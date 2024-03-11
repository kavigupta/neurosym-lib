from collections import defaultdict
from dataclasses import dataclass
from types import NoneType
from typing import List, Union

import numpy as np
import torch

from neurosym.dsl.dsl import DSL
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


@dataclass
class BigramProgramDistributionBatch:
    distribution_batch: np.ndarray

    def __post_init__(self):
        assert self.distribution_batch.ndim == 4
        assert self.distribution_batch.shape[1] == self.distribution_batch.shape[3]

    def __getitem__(self, i):
        return BigramProgramDistribution(self.distribution_batch[i])


@dataclass
class BigramProgramCountsTensorBatch:
    counts: torch.tensor

    def __post_init__(self):
        assert self.counts.ndim == 4
        assert self.counts.shape[1] == self.counts.shape[3]


class BigramProgramDistributionFamily(TreeProgramDistributionFamily):
    def __init__(self, dsl: DSL, valid_root_types: Union[NoneType, List[Type]] = None):
        self._dsl = dsl
        self._symbols, self._arities, self._valid_mask = bigram_mask(
            dsl, valid_root_types=valid_root_types
        )
        self._symbol_to_idx = {sym: i for i, sym in enumerate(self._symbols)}

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

    def count_programs(
        self, data: List[List[SExpression]]
    ) -> BigramProgramCountsTensorBatch:
        counts = np.zeros((len(data), *self.parameters_shape()), dtype=np.float32)
        for i, programs in enumerate(data):
            for program in programs:
                self._count_program(
                    program, counts, i, parent_sym=0, parent_child_idx=0
                )
        return BigramProgramCountsTensorBatch(torch.tensor(counts))

    def counts_to_distribution(
        self, counts: BigramProgramCountsTensorBatch
    ) -> BigramProgramDistributionBatch:
        return BigramProgramDistributionBatch(
            counts_to_probabilities(counts.counts.numpy())
        )

    def _count_program(
        self,
        program: SExpression,
        counts: np.ndarray,
        batch_idx: int,
        *,
        parent_sym: int,
        parent_child_idx: int,
    ):
        this_idx = self._symbol_to_idx[program.symbol]
        counts[batch_idx, parent_sym, parent_child_idx, this_idx] += 1
        for j, child in enumerate(program.children):
            self._count_program(
                child, counts, batch_idx, parent_sym=this_idx, parent_child_idx=j
            )
        return torch.tensor(counts)

    def parameter_difference_loss(
        self, parameters: torch.tensor, actual: BigramProgramCountsTensorBatch
    ) -> torch.float32:
        """
        E[log Q(|x)]
        """
        actual = actual.counts.to(parameters.device)
        parameters = self.normalize_parameters(parameters, logits=True, neg_inf=-100)
        combination = actual * parameters
        combination = combination.reshape(combination.shape[0], -1)
        return -combination.sum(-1)

    def uniform(self):
        return BigramProgramDistribution(counts_to_probabilities(self._valid_mask))

    def compute_tree_distribution(
        self, distribution: BigramProgramDistribution
    ) -> TreeDistribution:
        assert isinstance(distribution, BigramProgramDistribution)
        dist = defaultdict(list)
        for parent, position, child in zip(*np.where(distribution.distribution > 0)):
            dist[parent, position].append(
                (child, np.log(distribution.distribution[parent, position, child]))
            )
        dist = {k: sorted(v, key=lambda x: -x[1]) for k, v in dist.items()}

        return TreeDistribution(1, dist, list(zip(self._symbols, self._arities)))


def bigram_mask(dsl, valid_root_types: Union[NoneType, List[Type]] = None):
    symbols = ["<root>"] + sorted([x.symbol() for x in dsl.productions])

    if valid_root_types is None:
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
