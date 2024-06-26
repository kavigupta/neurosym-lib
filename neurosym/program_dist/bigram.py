from collections import defaultdict
from dataclasses import dataclass
from types import NoneType
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch

from neurosym.dsl.dsl import DSL, ROOT_SYMBOL
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
from neurosym.utils.documentation import internal_only

from .tree_distribution.tree_distribution import TreeProgramDistributionFamily


@dataclass
class BigramProgramDistribution:
    """
    Represents a bigram program distribution. This is a distribution over
    programs that is conditioned on the parent and the position of the child
    in the parent.

    :field dist_fam: The family of distributions that this distribution is from.
    :field distribution: A numpy array of shape ``(num_symbols, max_arity, num_symbols)``
        where ``num_symbols`` is the number of symbols in the DSL and ``max_arity`` is
        the maximum arity of any symbol. The value at ``(i, j, k)`` is the probability
        of symbol ``k`` appearing in position ``j`` of symbol ``i``.
    """

    dist_fam: "BigramProgramDistributionFamily"
    distribution: np.ndarray

    def __post_init__(self):
        assert self.distribution.ndim == 3
        assert self.distribution.shape[0] == self.distribution.shape[2]

    def bound_minimum_likelihood(
        self, min_likelihood: float, symbol_mask: np.ndarray = None
    ):
        """
        Ensure that the probability of any symbol is at least ``min_likelihood``,
        approximately, except symbols that have been masked out. This is done
        by setting the probability of any symbol that is not masked out to be at
        least ``min_likelihood``, then renormalizing the distribution.

        :param min_likelihood: The minimum likelihood that any symbol should have.
        :param symbol_mask: A boolean array of shape ``(num_symbols,)`` that is
            False for symbols that should not be affected by the minimum likelihood
            bound. If None, all symbols are affected.
        """
        assert 0 <= min_likelihood <= 1
        if symbol_mask is not None:
            assert (
                len(symbol_mask.shape) == 1
                and symbol_mask.shape[0] == self.distribution.shape[0]
            )
            assert symbol_mask.dtype == np.bool_
        distribution = self.distribution
        if symbol_mask is None:
            distribution = np.maximum(distribution, min_likelihood)
        else:
            mask_square = self._square_mask(symbol_mask)
            distribution = distribution.copy()
            distribution[mask_square] = np.maximum(
                distribution[mask_square], min_likelihood
            )
        distribution = self.dist_fam.mask_invalid(distribution)
        distribution = distribution / (distribution.sum(-1)[..., None] + 1e-10)
        return BigramProgramDistribution(self.dist_fam, distribution)

    def _square_mask(self, symbol_mask):
        mask_square = symbol_mask[:, None] & symbol_mask[None, :]
        mask_square = mask_square[:, None, :].repeat(self.distribution.shape[1], axis=1)

        return mask_square

    def mix_with_other(self, other: "BigramProgramDistribution", weight_other: float):
        """
        Mix this distribution with another distribution. This is done by taking
        a weighted average of the two distributions, where the weight of the other
        distribution is ``weight_other`` and the weight of this distribution is
        ``1 - weight_other``.
        """
        # pylint: disable=self-cls-assignment
        assert 0 <= weight_other <= 1
        symbols_this = self.dist_fam.symbols()
        symbols_other = other.dist_fam.symbols()
        if set(symbols_this).issubset(symbols_other):
            self, other = other, self
            symbols_this, symbols_other = symbols_other, symbols_this
            weight_other = 1 - weight_other
        else:
            if not set(symbols_this).issuperset(symbols_other):
                extra_this = set(symbols_this) - set(symbols_other)
                extra_other = set(symbols_other) - set(symbols_this)
                extra_this, extra_other = (
                    ", ".join(repr(x) for x in sorted(extra))
                    for extra in (extra_this, extra_other)
                )
                raise ValueError(
                    "DSL not compatible, extra symbols in this: "
                    f"{extra_this}, extra symbols in other: {extra_other}"
                )
        mask = np.array([s in symbols_other for s in symbols_this], dtype=np.bool_)
        distribution = self.distribution.copy()
        distribution_other = np.zeros_like(distribution)
        distribution_other[self._square_mask(mask)] = other.distribution.flatten()
        distribution[mask] = (
            weight_other * distribution_other[mask]
            + (1 - weight_other) * distribution[mask]
        )
        return BigramProgramDistribution(self.dist_fam, distribution)


@dataclass
class _BigramProgramDistributionBatch:
    dist_fam: "BigramProgramDistributionFamily"
    distribution_batch: np.ndarray

    def __post_init__(self):
        assert isinstance(self.distribution_batch, np.ndarray), type(
            self.distribution_batch
        )
        assert self.distribution_batch.ndim == 4
        assert self.distribution_batch.shape[1] == self.distribution_batch.shape[3]

    def __getitem__(self, i):
        return BigramProgramDistribution(self.dist_fam, self.distribution_batch[i])

    def __len__(self):
        return len(self.distribution_batch)


@dataclass
class BigramProgramCounts:
    """
    Represents counts of bigram programs, both the "numerator" counts
        (the number of times a symbol appears in a given context), and
        the "denominator" counts (the number of times a symbol could
        have appeared in that context).

    :field numerators: A map from context ``(parent_sym, parent_child_idx)`` to a map from
        ``child_sym`` to ``count``.
    :field denominators: A map from context ``(parent_sym, parent_child_idx)`` to a map from
        ``child_syms`` to ``count``. The ``child_syms`` are the set of symbols that could have
        appeared in that context.
    """

    # map from (parent_sym, parent_child_idx) to map from child_sym to count
    numerators: Dict[Tuple[Tuple[int, int], ...], Dict[int, int]]
    # map from (parent_sym, parent_child_idx) to map from potential child_sym values to count
    denominators: Dict[Tuple[Tuple[int, int], ...], Dict[Tuple[int, ...], int]]

    def add_to_numerator_array(self, arr, batch_idx):
        """
        Add the numerator counts to the given array at the given batch index.

        :param arr: An array of counts, to be mutated. Must be indexable as
            ``arr[batch_idx, parent_sym, parent_child_idx, child_sym]``.
            A reference to this array is returned.
        :param batch_idx: The batch index to add the counts to.
        """
        for [(parent_sym, parent_child_idx)], children in self.numerators.items():
            for child_sym, count in children.items():
                arr[batch_idx, parent_sym, parent_child_idx, child_sym] = count
        return arr

    def add_to_denominator_array(self, arr, batch_idx, backmap):
        """
        Add the denominator counts to the given array at the given batch index.

        :param arr: An array of counts, to be mutated. Must be indexable as
            ``arr[batch_idx, parent_sym, parent_child_idx, denominator_id]``.
            A reference to this array is returned.
        :param batch_idx: The batch index to add the counts to.
        :param backmap: A mapping from the set of child symbols to the index in the
            denominator array.
        """
        for [(parent_sym, parent_child_idx)], children in self.denominators.items():
            for child_syms, count in children.items():
                key = batch_idx, parent_sym, parent_child_idx, backmap[child_syms]
                arr[key] = count
        return arr


@dataclass
class BigramProgramCountsBatch:
    """
    Like BigramProgramCounts, but batched, that is, it contains a list of
    BigramProgramCounts objects.

    :field dist_fam: The family of distributions that these counts are for.
    :field counts: A list of BigramProgramCounts objects.
    """

    dist_fam: "BigramProgramDistributionFamily"
    counts: List[BigramProgramCounts]

    def numerators(self, num_symbols, max_arity):
        """
        See ``BigramProgramCounts.numerators`` for details. This is a batched version
        that creates and returns a numpy array of counts.
        """
        numerators = np.zeros(
            (len(self.counts), num_symbols, max_arity, num_symbols), dtype=np.int32
        )
        for i, dist in enumerate(self.counts):
            dist.add_to_numerator_array(numerators, i)
        return numerators

    def denominators(self, num_symbols, max_arity):
        """
        See ``BigramProgramCounts.denominators`` for details. This is a batched version
        that creates and returns a numpy array of counts, along with the denominator index
        sets that correspond to the last axis of the array.
        """
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
        """
        Convert these counts to a ``BigramProgramDistribution`` object. This is a
        batched version that creates and returns a ``BigramProgramDistributionBatch`` object.

        Note that we are not handling denominators here, as this is just a simple conversion
        from counts to probabilities. This is probably not fully correct, as we should be
        using a more sophisticated algorithm to fit the parameters to the counts. However,
        this is a relatively simple way to get a distribution from counts.
        """
        # TODO(kavigupta): Implement a proper fitting algorithm here.
        numerators = self.numerators(num_symbols, max_arity)

        return _BigramProgramDistributionBatch(
            self.dist_fam, _counts_to_probabilities(numerators)
        )


class BigramProgramDistributionFamily(TreeProgramDistributionFamily):
    """
    A family of bigram program distributions. These are TreeProgramDistributions
    that are conditioned on just the parent and the position of the child
    in the parent. This is a kind of TreeProgramDistributionFamily.

    :param dsl: The DSL that this family is for.
    :param valid_root_types: The types that are valid as roots of programs.
    :param additional_preorder_masks: A tuple of functions that take a TreeDistribution
        and return a PreorderMask. These masks are used to filter the set of possible
        symbols that can appear in a given context.
    :param include_type_preorder_mask: Whether to include a type preorder mask in the
        set of masks that are used to filter the set of possible symbols that can appear
        in a given context.
    :param node_ordering: The node ordering to use when traversing the tree. This
        determines the order in which the children of a node are considered.
    """

    def __init__(
        self,
        dsl: DSL,
        valid_root_types: Union[NoneType, List[Type]] = None,
        *,
        additional_preorder_masks: Tuple[
            Callable[[DSL, TreeDistribution], PreorderMask]
        ] = (),
        include_type_preorder_mask: bool = True,
        node_ordering=DefaultNodeOrdering,
    ):
        if valid_root_types is not None:
            dsl = dsl.with_valid_root_types(valid_root_types)
        self._dsl = dsl
        self._symbols, self._arities, self._valid_mask = _bigram_mask(dsl)
        self._max_arity = max(self._arities)
        self._symbol_to_idx = {sym: i for i, sym in enumerate(self._symbols)}
        self._additional_preorder_masks = additional_preorder_masks
        self._include_type_preorder_mask = include_type_preorder_mask
        self._node_ordering = node_ordering

    def underlying_dsl(self) -> DSL:
        return self._dsl

    def parameters_shape(self) -> List[int]:
        return self._valid_mask.shape

    def _normalize_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Apply a mask to the parameters, then normalize them to be a valid
        probability distribution. If ``logits`` is True, the parameters are
        returned as logits, otherwise they are returned as probabilities.
        """
        parameters = parameters.clone()
        mask = torch.tensor(self._valid_mask, device=parameters.device)[None].repeat(
            parameters.shape[0], 1, 1, 1
        )
        parameters[~mask] = -float("inf")
        parameters = parameters.softmax(-1)
        parameters[~mask] = 0
        return parameters

    def with_parameters(
        self, parameters: torch.Tensor
    ) -> _BigramProgramDistributionBatch:
        assert (
            parameters.shape[1:] == self.parameters_shape()
        ), f"Expected {self.parameters_shape()}, got {parameters.shape[1:]}"
        parameters = self._normalize_parameters(parameters)
        return _BigramProgramDistributionBatch(self, parameters.detach().cpu().numpy())

    def count_programs(self, data: List[List[SExpression]]) -> BigramProgramCountsBatch:
        tree_dist = self.tree_distribution_skeleton
        all_counts = []
        for programs in data:
            numerators, denominators = _count_programs(tree_dist, programs)
            all_counts.append(
                BigramProgramCounts(numerators=numerators, denominators=denominators)
            )
        return BigramProgramCountsBatch(self, all_counts)

    def counts_to_distribution(
        self, counts: BigramProgramCountsBatch
    ) -> BigramProgramDistribution:
        return counts.to_distribution(len(self._symbols), self._max_arity)

    def parameter_difference_loss(
        self, parameters: torch.tensor, actual: BigramProgramCountsBatch
    ) -> torch.float32:
        # Let
        #     p be a program
        #     g be a bigram context (i.e. parent +, in position 2)
        #     s be a symbol
        #     d be the "denominator" (which is the set of *possible* symbols s'
        #         that could have appeared in this context g).
        # The ngram ``g`` is really the context that ``s`` appears in. So for example
        #     ``(+ (- 1 2) 3)`` if we're thinking about the symbol s=``-`` its bigram g=``(+, 0)``
        #     indicating it's the first argument of a "+". This is the bigram notion from
        #     DreamCoder where it's not just the parent (analogous to "previous token" in NLP)
        #     but also includes which child of the parent we are. In general, we also filter
        #     on other characteristics (e.g., type checking), not just ngrams. See the
        #     preorder mask for more details.

        # In this particular example, ``d`` would be the set of symbols that *could* have
        #     appeared in that same position ``g``, for example ``1`` or ``+`` or ``-`` or anything
        #     else that type checks. This is called the denominator because the
        #     probability of choosing ``s`` among ``(s' in d)`` is going to be
        #         P(s)/sum_{s' in d} P(s')
        #     These probabilities will depend on the parameters ``theta`` of the bigram model
        #     (and of course, specifically the unigram it assigns to the context ``g``).

        # (See also math below) Our goal is to compute the loglikelihood of the actual
        #     program P under the bigram parameters theta. Which, for a bigram is the sum
        #     of the logprob over all subtrees of the symbol ``s`` context ``g`` and denominator ``d`` for
        #     that subtree. We can factor this overall sum into two parts: a numerator based on
        #     sthe actual symbol ``s`` and a denominator based on the alternative symbols in ``d``
        #     for that context ``g``.

        # sum_p log P(p | theta)
        #     = sum_p sum_{(g, s, d) in p} log P(s | g, theta, d)
        #     = sum_p sum_{(g, s, d) in p} log (exp(theta_{g, s}) / sum_{s' in d} exp(theta_{g, s'}))
        #     = sum_p sum_{(g, s, d) in p} (theta_{g, s} - log sum_{s' in d} exp(theta_{g, s'}))
        #     = [sum_p sum_{(g, s, d) in p} theta_{g, s}]
        #         - [sum_p sum_{(g, s, d) in p} log sum_{s' in d} exp(theta_{g, s'})]
        #     = [numer] - [denom]

        # For each (g,s) instance in the corpus the numerator is the same, and for each (g,d)
        #     instance the denominator is the same, so we can instead come up with counts for
        #     each of these (calling them ``numcount`` and ``dencount``) and rewrite our sum as a
        #     sum over the unique (g,s) and (g,d) instances:

        # numer
        #     = sum_p sum_{(g, s, d) in p} theta_{g, s}
        #     = sum_g sum_s numcount_{g, s} theta_{g, s}
        #     = (numcount * theta).sum()

        # denom
        #     = sum_p sum_{(g, s, d) in p} log sum_{s' in d} exp(theta_{g, s'})
        #     = sum_g sum_d dencount_{g, d} log sum_{s' in d} exp(theta_{g, s'})

        # theta_by_denom(g, s', d) = theta_{g, s'} if s' in d, else -inf

        # denom
        #     = sum_g sum_d dencount_{g, d} log sum_{s'} exp(theta_by_denom{g, s', d})

        # agg_theta_by_denom(g, d) = logsumexp(theta_by_denom(g, *, d))

        # denom
        #     = sum_g sum_d dencount_{g, d} agg_theta_by_denom(g, d)
        #     = (dencount * agg_theta_by_denom).sum()

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
        """
        A PCFG which is "uniform" in the sense that given a parent and a position
        in the parent, all symbols are equally likely. This distribution is not
        necessarily a proper probability distribution, as many paths suggested
        by this distribution will be invalid. However, it can be enumerated.
        """
        return BigramProgramDistribution(
            self, _counts_to_probabilities(self._valid_mask)
        )

    def compute_tree_distribution(
        self, distribution: Union[BigramProgramDistribution, NoneType]
    ) -> TreeDistribution:
        if isinstance(distribution, BigramProgramDistribution):
            assert isinstance(distribution, BigramProgramDistribution), type(
                distribution
            )
            dist_vals = distribution.distribution
        else:
            assert distribution is None
            dist_vals = self._valid_mask

        dist = defaultdict(list)
        for parent, position, child in zip(*np.where(dist_vals > 0)):
            dist[(parent, position),].append(
                (child, np.log(dist_vals[parent, position, child]))
            )
        dist = {k: sorted(v, key=lambda x: -x[1]) for k, v in dist.items()}

        return TreeDistribution(
            1,
            dist,
            list(zip(self._symbols, self._arities)),
            self._compute_preorder_mask,
            self._node_ordering,
        )

    def _compute_preorder_mask(self, tree_dist):
        masks = []
        if self._include_type_preorder_mask:
            masks.append(TypePreorderMask(tree_dist, self._dsl))
        for mask in self._additional_preorder_masks:
            masks.append(mask(tree_dist, self._dsl))
        return ConjunctionPreorderMask.of(tree_dist, masks)

    @internal_only
    def mask_invalid(self, distribution):
        return distribution * self._valid_mask

    def symbols(self) -> List[str]:
        """
        Get the symbols this distribution is over.
        """
        return self._symbols


def _bigram_mask(dsl):
    symbols = dsl.ordered_symbols(include_root=True)

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
        valid_mask[symbol_to_idx[ROOT_SYMBOL], 0, symbol_to_idx[root_sym]] = 1
    for _, rules in rules_for.items():
        for root_sym, child_types in rules:
            for i, ct in enumerate(child_types):
                for child_sym, _ in rules_for.get(ct, ()):
                    valid_mask[symbol_to_idx[root_sym], i, symbol_to_idx[child_sym]] = 1
    return symbols, np.array(arities), valid_mask


def _counts_to_probabilities(counts):
    return np.divide(
        counts,
        counts.sum(-1)[..., None],
        out=np.zeros_like(counts, dtype=np.float32),
        where=counts != 0,
    )


def _count_programs(tree_dist: TreeDistribution, programs: List[SExpression]):
    """
    Count the productions in the programs, indexed by the path to the node.
    """
    numerators = defaultdict(lambda: defaultdict(int))
    denominators = defaultdict(lambda: defaultdict(int))
    for program in programs:
        preorder_mask = tree_dist.mask_constructor(tree_dist)
        preorder_mask.on_entry(0, 0)
        _accumulate_counts(
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


def _accumulate_counts(
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
        _accumulate_counts(
            tree_dist,
            child,
            numerators,
            denominators,
            new_ancestors,
            preorder_mask=preorder_mask,
        )
    preorder_mask.on_exit(parent_position, this_idx)
