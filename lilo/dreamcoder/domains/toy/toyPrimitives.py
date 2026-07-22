from dreamcoder.program import Primitive
from dreamcoder.type import arrow, tint

def _incr(x): return x + 1
def _incr2(x): return x + 2

def toyPrimitives():
    return [
        Primitive("1", tint, 1),
        Primitive("incr", arrow(tint, tint), _incr),
        Primitive("incr2", arrow(tint, tint), _incr2),
    ]