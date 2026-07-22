import itertools
import unittest
from fractions import Fraction

import numpy as np
from utils import enumerate_dsl

import neurosym as ns

if __name__ == "__main__":
    dslf = ns.DSLFactory()
    dslf.concrete("1", "() -> i", lambda: 1)
    dslf.concrete("incr", "(i) -> ii", lambda x: x + 1)
    dslf.concrete("incr2", "(i) -> ii", lambda x: x + 2)
    dslf.prune_to("ii")
    dsl = dslf.finalize()
    family = ns.BigramProgramDistributionFamily(dsl)
    dist = family.uniform()
    print(enumerate_dsl(family, dist, min_likelihood=-1000000))
    concrete_prods = dslf._parameterized_productions
    symbol_list = []
    sig_list = []
    semantics_list = []
    for prods in concrete_prods:
        symbol, sig, semantics, _ = prods
        symbol_list.append(symbol)
        sig_list.append(sig)
        semantics_list.append(semantics)
