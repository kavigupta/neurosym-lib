from dataclasses import dataclass
from typing import List

import numpy as np

from neurosym.programs.s_expression import SExpression


@dataclass
class PCFGPattern:
    symbols: List[str]
    arities: List[int]
    valid_mask: np.ndarray  # (num_symbols, max_arity, num_symbols)

    @classmethod
    def of(cls, dsl, *out_ts):
        symbols = ["<root>"] + [x.symbol() for x in dsl.productions]
        symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}
        rules_for = dsl.all_rules(
            *out_ts, care_about_variables=False, type_depth_limit=float("inf")
        )
        arities = [None] * len(symbols)
        arities[0] = 1
        for _, rules in rules_for.items():
            for sym, children in rules:
                assert arities[symbol_to_idx[sym]] is None
                arities[symbol_to_idx[sym]] = len(children)
        assert all(
            x is not None for x in arities
        ), f"No arities for symbols {[sym for sym, ar in zip(symbols, arities) if ar is None]}"
        valid_mask = np.zeros(
            (len(symbols), max(arities), len(symbols)), dtype=np.bool_
        )
        root_syms = {sym for t in out_ts for sym, _ in rules_for[t]}
        for root_sym in root_syms:
            valid_mask[symbol_to_idx["<root>"], 0, symbol_to_idx[root_sym]] = 1
        for _, rules in rules_for.items():
            for root_sym, child_types in rules:
                for i, ct in enumerate(child_types):
                    for child_sym, _ in rules_for[ct]:
                        valid_mask[
                            symbol_to_idx[root_sym], i, symbol_to_idx[child_sym]
                        ] = 1
        return cls(symbols, np.array(arities), valid_mask)

    def uniform(self):
        return PCFG(
            self.symbols, self.arities, self._convert_counts_to_probs(self.valid_mask)
        )

    def _convert_counts_to_probs(self, counts):
        parent_idxs, loc_idxs = np.arange(counts.shape[0]), np.arange(counts.shape[1])
        parent_idxs, loc_idxs = np.meshgrid(parent_idxs, loc_idxs, indexing="ij")
        update_mask = loc_idxs < self.arities[:, None]
        probs = np.zeros_like(counts, dtype=np.float64)
        probs[update_mask] = (
            counts[update_mask] / counts[update_mask].sum(-1)[..., None]
        )
        return probs

    def count(self, programs):
        counts = np.zeros_like(self.valid_mask, dtype=np.int64)
        for program in programs:
            self._count_program(program, counts)
        return counts

    def _count_program(self, program, counts, *, parent_sym="<root>", loc=0):
        if program.symbol not in self.symbols:
            return
        sym_idx = self.symbols.index(program.symbol)
        counts[self.symbols.index(parent_sym), loc, sym_idx] += 1
        for i, child in enumerate(program.children):
            self._count_program(child, counts, parent_sym=program.symbol, loc=i)


@dataclass
class PCFG:
    symbols: List[str]
    arities: List[int]
    probabilities: np.ndarray  # (num_symbols, max_arity, num_symbols)

    def _sample(self, rng, symbol_idx, max_depth):
        if max_depth == 0:
            raise TooDeepError()
        arity = self.arities[symbol_idx]
        result = []
        for i in range(arity):
            probs = self.probabilities[symbol_idx, i]
            child_idx = rng.choice(np.arange(len(self.symbols)), p=probs)
            result.append(self._sample(rng, child_idx, max_depth - 1))
        return SExpression(self.symbols[symbol_idx], tuple(result))

    def sample(self, rng, max_depth):
        while True:
            try:
                return self._sample(rng, 0, max_depth).children[0]
            except TooDeepError:
                pass


class TooDeepError(Exception):
    pass
