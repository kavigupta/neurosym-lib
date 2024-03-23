from collections import defaultdict
from dataclasses import dataclass
from types import NoneType
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch

from neurosym.dsl.dsl import DSL
from neurosym.program_dist.tree_distribution.ordering import DefaultNodeOrdering
from neurosym.program_dist.tree_distribution.preorder_mask.preorder_mask import (
    ConjunctionPreorderMask,
    PreorderMask,
)
from neurosym.program_dist.tree_distribution.preorder_mask.type_preorder_mask import (
    TypePreorderMask,
)
from neurosym.program_dist.tree_distribution.tree_distribution import TreeDistribution
from neurosym.programs.s_expression import SExpression
from neurosym.types.type import Type

from .tree_distribution.tree_distribution import TreeProgramDistributionFamily


@dataclass
class BigramProgramDistribution:
    distribution: np.ndarray

    def __post_init__(self):
        assert self.distribution.ndim == 3
        assert self.distribution.shape[0] == self.distribution.shape[2]

    def bound_minimum_likelihood(self, min_likelihood: float):
        assert 0 <= min_likelihood <= 1
        distribution = self.distribution
        distribution = np.maximum(distribution, min_likelihood)
        distribution = distribution / distribution.sum(-1)[..., None]
        return BigramProgramDistribution(distribution)


@dataclass
class BigramProgramDistributionBatch:
    distribution_batch: np.ndarray

    def __post_init__(self):
        assert isinstance(self.distribution_batch, np.ndarray), type(
            self.distribution_batch
        )
        assert self.distribution_batch.ndim == 4
        assert self.distribution_batch.shape[1] == self.distribution_batch.shape[3]

    def __getitem__(self, i):
        return BigramProgramDistribution(self.distribution_batch[i])

    def __len__(self):
        return len(self.distribution_batch)


@dataclass
class BigramProgramCounts:
    # map from (parent_sym, parent_child_idx) to map from child_sym to count
    numerators: Dict[Tuple[Tuple[int, int], ...], Dict[int, int]]
    # map from (parent_sym, parent_child_idx) to map from potential child_sym values to count
    denominators: Dict[Tuple[Tuple[int, int], ...], Dict[Tuple[int, ...], int]]

    def add_to_numerator_array(self, arr, batch_idx):
        for [(parent_sym, parent_child_idx)], children in self.numerators.items():
            for child_sym, count in children.items():
                arr[batch_idx, parent_sym, parent_child_idx, child_sym] = count
        return arr

    def add_to_denominator_array(self, arr, batch_idx, backmap):
        for [(parent_sym, parent_child_idx)], children in self.denominators.items():
            for child_syms, count in children.items():
                key = batch_idx, parent_sym, parent_child_idx, backmap[child_syms]
                arr[key] = count
        return arr


@dataclass
class BigramProgramCountsBatch:
    counts: List[BigramProgramCounts]

    def numerators(self, num_symbols, max_arity):
        numerators = np.zeros(
            (len(self.counts), num_symbols, max_arity, num_symbols), dtype=np.int32
        )
        for i, dist in enumerate(self.counts):
            dist.add_to_numerator_array(numerators, i)
        return numerators

    def denominators(self, num_symbols, max_arity):
        denominator_keys = {
            key
            for counts in self.counts
            for mapping in counts.denominators.values()
            for key in mapping.keys()
        }
        denominator_keys = sorted(denominator_keys)
        denominator_keys_backmap = {key: i for i, key in enumerate(denominator_keys)}
        denominators = np.zeros(
            (len(self.counts), num_symbols, max_arity, len(denominator_keys)),
            dtype=np.int32,
        )
        for i, counts in enumerate(self.counts):
            counts.add_to_denominator_array(denominators, i, denominator_keys_backmap)

        return denominators, denominator_keys

    def to_distribution(self, num_symbols, max_arity):
        numerators = self.numerators(num_symbols, max_arity)

        # We do not need to handle denominators here, as this is just
        # a simple conversion from counts to probabilities, and we do
        # not need to handle the case where the denominator is 0.

        return BigramProgramDistributionBatch(counts_to_probabilities(numerators))


class BigramProgramDistributionFamily(TreeProgramDistributionFamily):
    def __init__(
        self,
        dsl: DSL,
        valid_root_types: Union[NoneType, List[Type]] = None,
        *,
        additional_preorder_masks: Tuple[
            Callable[[DSL, TreeDistribution], PreorderMask]
        ] = (),
        include_type_preorder_mask: bool = True,
        node_ordering=lambda _: DefaultNodeOrdering(),
    ):
        if valid_root_types is not None:
            dsl = dsl.with_valid_root_types(valid_root_types)
        self._dsl = dsl
        self._symbols, self._arities, self._valid_mask = bigram_mask(dsl)
        self._max_arity = max(self._arities)
        self._symbol_to_idx = {sym: i for i, sym in enumerate(self._symbols)}
        self._additional_preorder_masks = additional_preorder_masks
        self._include_type_preorder_mask = include_type_preorder_mask
        self._node_ordering = node_ordering

    def underlying_dsl(self) -> DSL:
        return self._dsl

    def parameters_shape(self) -> List[int]:
        return self._valid_mask.shape

    def normalize_parameters(
        self, parameters: torch.Tensor, *, logits: bool, neg_inf=-float("inf")
    ) -> torch.Tensor:
        parameters = parameters.clone()
        mask = torch.tensor(self._valid_mask, device=parameters.device)[None].repeat(
            parameters.shape[0], 1, 1, 1
        )
        parameters[~mask] = -float("inf")
        if logits:
            parameters = parameters.log_softmax(-1)
            parameters[~mask] = neg_inf
        else:
            parameters = parameters.softmax(-1)
            parameters[~mask] = 0
        return parameters

    def with_parameters(
        self, parameters: torch.Tensor
    ) -> BigramProgramDistributionBatch:
        assert (
            parameters.shape[1:] == self.parameters_shape()
        ), f"Expected {self.parameters_shape()}, got {parameters.shape[1:]}"
        parameters = self.normalize_parameters(parameters, logits=False)
        return BigramProgramDistributionBatch(parameters.detach().cpu().numpy())

    def count_programs(self, data: List[List[SExpression]]) -> BigramProgramCountsBatch:
        tree_dist = self.tree_distribution_skeleton
        all_counts = []
        for programs in data:
            numerators, denominators = count_programs(tree_dist, programs)
            all_counts.append(
                BigramProgramCounts(numerators=numerators, denominators=denominators)
            )
        return BigramProgramCountsBatch(all_counts)

    def counts_to_distribution(
        self, counts: BigramProgramCountsBatch
    ) -> BigramProgramDistribution:
        return counts.to_distribution(len(self._symbols), self._max_arity)

    def parameter_difference_loss(
        self, parameters: torch.tensor, actual: BigramProgramCountsBatch
    ) -> torch.float32:
        """
        Let
            p be a program
            g be a bigram context (i.e. parent +, in position 2)
            s be a symbol
            d be the "denominator" (which is the set of *possible* symbols s'
                that could have appeared in this context g).
        The ngram `g` is really the context that `s` appears in. So for example
            `(+ (- 1 2) 3)` if we're thinking about the symbol s=`-` its bigram g=`(+, 0)`
            indicating it's the first argument of a "+". This is the bigram notion from
            DreamCoder where it's not just the parent (analogous to "previous token" in NLP)
            but also includes which child of the parent we are. In general, we also filter
            on other characteristics (e.g., type checking), not just ngrams. See the
            preorder mask for more details.

        In this particular example, `d` would be the set of symbols that *could* have
            appeared in that same position `g`, for example `1` or `+` or `-` or anything
            else that type checks. This is called the denominator because the
            probability of choosing `s` among `(s' in d)` is going to be
                P(s)/sum_{s' in d} P(s')
            These probabilities will depend on the parameters `theta` of the bigram model
            (and of course, specifically the unigram it assigns to the context `g`).

        (See also math below) Our goal is to compute the loglikelihood of the actual
            program P under the bigram parameters theta. Which, for a bigram is the sum
            of the logprob over all subtrees of the symbol `s` context `g` and denominator `d` for
            that subtree. We can factor this overall sum into two parts: a numerator based on
            sthe actual symbol `s` and a denominator based on the alternative symbols in `d`
            for that context `g`.

        sum_p log P(p | theta)
            = sum_p sum_{(g, s, d) in p} log P(s | g, theta, d)
            = sum_p sum_{(g, s, d) in p} log (exp(theta_{g, s}) / sum_{s' in d} exp(theta_{g, s'}))
            = sum_p sum_{(g, s, d) in p} (theta_{g, s} - log sum_{s' in d} exp(theta_{g, s'}))
            = [sum_p sum_{(g, s, d) in p} theta_{g, s}]
                - [sum_p sum_{(g, s, d) in p} log sum_{s' in d} exp(theta_{g, s'})]
            = [numer] - [denom]

        For each (g,s) instance in the corpus the numerator is the same, and for each (g,d)
            instance the denominator is the same, so we can instead come up with counts for
            each of these (calling them `numcount` and `dencount`) and rewrite our sum as a
            sum over the unique (g,s) and (g,d) instances:

        numer
            = sum_p sum_{(g, s, d) in p} theta_{g, s}
            = sum_g sum_s numcount_{g, s} theta_{g, s}
            = (numcount * theta).sum()

        denom
            = sum_p sum_{(g, s, d) in p} log sum_{s' in d} exp(theta_{g, s'})
            = sum_g sum_d dencount_{g, d} log sum_{s' in d} exp(theta_{g, s'})

        theta_by_denom(g, s', d) = theta_{g, s'} if s' in d, else -inf

        denom
            = sum_g sum_d dencount_{g, d} log sum_{s'} exp(theta_by_denom{g, s', d})

        agg_theta_by_denom(g, d) = logsumexp(theta_by_denom(g, *, d))

        denom
            = sum_g sum_d dencount_{g, d} agg_theta_by_denom(g, d)
            = (dencount * agg_theta_by_denom).sum()

        """

        assert parameters.shape[1:] == self.parameters_shape()
        assert len(parameters.shape) == 4

        numcount = actual.numerators(len(self._symbols), self._max_arity)
        dencount, den_keys = actual.denominators(len(self._symbols), self._max_arity)
        numcount, dencount = [
            torch.tensor(x, device=parameters.device, dtype=torch.float32)
            for x in (numcount, dencount)
        ]

        def agg_across_all_but_batch_axis(x, fn):
            x = x.reshape(x.shape[0], -1)
            return fn(x, -1)

        numer = agg_across_all_but_batch_axis(numcount * parameters, torch.sum)

        theta_by_denom = parameters[..., None].repeat(1, 1, 1, 1, len(den_keys))
        for i, key in enumerate(den_keys):
            mask = torch.ones(
                len(self._symbols), dtype=torch.bool, device=parameters.device
            )
            mask[list(key)] = False
            theta_by_denom[..., mask, i] = -float("inf")
        agg_theta_by_denom = torch.logsumexp(theta_by_denom, dim=-2)
        denom = agg_across_all_but_batch_axis(dencount * agg_theta_by_denom, torch.sum)
        return -(numer - denom)

    def uniform(self):
        return BigramProgramDistribution(counts_to_probabilities(self._valid_mask))

    def compute_tree_distribution(
        self, distribution: Union[BigramProgramDistribution, NoneType]
    ) -> TreeDistribution:
        if isinstance(distribution, BigramProgramDistribution):
            assert isinstance(distribution, BigramProgramDistribution), type(
                distribution
            )
            dist = defaultdict(list)
            for parent, position, child in zip(
                *np.where(distribution.distribution > 0)
            ):
                dist[(parent, position),].append(
                    (child, np.log(distribution.distribution[parent, position, child]))
                )
            dist = {k: sorted(v, key=lambda x: -x[1]) for k, v in dist.items()}
        else:
            assert distribution is None
            dist = None

        return TreeDistribution(
            1,
            dist,
            list(zip(self._symbols, self._arities)),
            self.compute_preorder_mask,
            self._node_ordering,
        )

    def compute_preorder_mask(self, tree_dist):
        masks = []
        if self._include_type_preorder_mask:
            masks.append(TypePreorderMask(tree_dist, self._dsl))
        for mask in self._additional_preorder_masks:
            masks.append(mask(tree_dist, self._dsl))
        return ConjunctionPreorderMask(tree_dist, masks)


def bigram_mask(dsl):
    symbols = ["<root>"] + sorted([x.symbol() for x in dsl.productions])

    valid_root_types = dsl.valid_root_types
    symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}
    rules_for = dsl.all_rules(
        care_about_variables=False, valid_root_types=valid_root_types
    )
    arities = [None] * len(symbols)
    arities[0] = 1
    for _, rules in rules_for.items():
        for sym, children in rules:
            assert arities[symbol_to_idx[sym]] in (None, len(children)), str(
                symbol_to_idx
            )
            arities[symbol_to_idx[sym]] = len(children)
    assert all(
        x is not None for x in arities
    ), f"No arities for symbols {[sym for sym, ar in zip(symbols, arities) if ar is None]}"
    valid_mask = np.zeros((len(symbols), max(arities), len(symbols)), dtype=np.bool_)
    root_syms = {sym for t in valid_root_types for sym, _ in rules_for[t]}
    for root_sym in root_syms:
        valid_mask[symbol_to_idx["<root>"], 0, symbol_to_idx[root_sym]] = 1
    for _, rules in rules_for.items():
        for root_sym, child_types in rules:
            for i, ct in enumerate(child_types):
                for child_sym, _ in rules_for.get(ct, ()):
                    valid_mask[symbol_to_idx[root_sym], i, symbol_to_idx[child_sym]] = 1
    return symbols, np.array(arities), valid_mask


def counts_to_probabilities(counts):
    return np.divide(
        counts,
        counts.sum(-1)[..., None],
        out=np.zeros_like(counts, dtype=np.float32),
        where=counts != 0,
    )


def count_programs(tree_dist: TreeDistribution, programs: List[SExpression]):
    """
    Count the productions in the programs, indexed by the path to the node.
    """
    numerators = defaultdict(lambda: defaultdict(int))
    denominators = defaultdict(lambda: defaultdict(int))
    for program in programs:
        preorder_mask = tree_dist.mask_constructor(tree_dist)
        preorder_mask.on_entry(0, 0)
        accumulate_counts(
            tree_dist,
            program,
            numerators,
            denominators,
            ((0, 0),),
            preorder_mask=preorder_mask,
        )
    numerators = {k: dict(v) for k, v in numerators.items()}
    denominators = {k: dict(v) for k, v in denominators.items()}
    return numerators, denominators


def accumulate_counts(
    tree_dist: TreeDistribution,
    program: SExpression,
    numerators: Dict[Tuple[Tuple[int, int], ...], Dict[int, int]],
    denominators: Dict[Tuple[Tuple[int, int], ...], Dict[Tuple[int, ...], int]],
    ancestors: Tuple[Tuple[int, int], ...],
    preorder_mask: PreorderMask,
):
    parent_position = ancestors[-1][1]
    this_idx = tree_dist.symbol_to_index[program.symbol]
    numerators[ancestors][this_idx] += 1
    possibilities = np.arange(len(tree_dist.symbols))
    mask = preorder_mask.compute_mask(parent_position, possibilities)
    preorder_mask.on_entry(parent_position, this_idx)
    elements = possibilities[mask]
    elements = tuple(int(x) for x in elements)
    denominators[ancestors][elements] += 1
    order = tree_dist.ordering.order(this_idx, len(program.children))
    for j, child in zip(order, [program.children[i] for i in order]):
        new_ancestors = ancestors + ((this_idx, j),)
        new_ancestors = new_ancestors[-tree_dist.limit :]
        accumulate_counts(
            tree_dist,
            child,
            numerators,
            denominators,
            new_ancestors,
            preorder_mask=preorder_mask,
        )
    preorder_mask.on_exit(parent_position, this_idx)
