from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class PCFGPattern:
    symbols: List[str]
    valid_mask: np.ndarray  # (num_symbols, max_arity, num_symbols)

    @classmethod
    def of(cls, dsl, *out_ts):
        symbols = [x.symbol() for x in dsl.productions]
        symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}
        rules_for = dsl.all_rules(*out_ts)
        max_arity = max(
            len(children) for rules in rules_for.values() for _, children in rules
        )
        valid_mask = np.zeros((len(symbols), max_arity, len(symbols)), dtype=np.bool_)
        for _, rules in rules_for.items():
            for root_sym, child_types in rules:
                for i in range(len(child_types)):
                    for child_sym, _ in rules_for[child_types[i]]:
                        valid_mask[
                            symbol_to_idx[root_sym], i, symbol_to_idx[child_sym]
                        ] = 1
        return cls(symbols, valid_mask)
