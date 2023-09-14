from functools import reduce
from neurosym.dsl.dsl_factory import DSLFactory


dslf = DSLFactory()

for i in range(6):
    dslf.concrete(str(i), f"() -> i", lambda: i)

types = "i", "[i]", "[[i]]"
for T in types:
    dslf.concrete(f"empty_{T}", f"() -> [{T}]", lambda: [])
    dslf.concrete(f"singleton_{T}", f"{T} -> [{T}]", lambda x: [x])
dslf.concrete("range", f"i -> [i]", lambda x: list(range(x)))
for T in types:
    dslf.concrete(f"++_{T}", f"([{T}], [{T}]) -> [{T}]", lambda x, y: x + y)
    for R in types:
        dslf.concrete(
            f"mapi_{T}_{R}",
            f"((i, {T}) -> {R}, [{T}]) -> [{R}]",
            lambda f: lambda x: [f(i, x) for i, x in enumerate(x)],
        )
        dslf.concrete(
            f"reducei_{T}_{R}",
            f"((i, {R}, {T}) -> {R}, {R}, [{T}]) -> {R}",
            lambda f: lambda x: lambda y: reduce(lambda x, y: f(i, x, y), x, y),
        )
dslf.concrete("true", f"() -> b", lambda: True)
dslf.concrete("not", f"b -> b", lambda x: not x)
dslf.concrete("and", f"(b, b) -> b", lambda x, y: x and y)
dslf.concrete("or", f"(b, b) -> b", lambda x, y: x or y)
for T in types:
    dslf.concrete(f"if_{T}", f"(b, {T}, {T}) -> {T}", lambda x, y, z: y if x else z)
    dslf.concrete(f"sort_{T}", f"([{T}]) -> [{T}]", lambda x: sorted(x))
dslf.concrete("+", f"(i, i) -> i", lambda x, y: x + y)
dslf.concrete("*", f"(i, i) -> i", lambda x, y: x * y)
dslf.concrete("negate", f"i -> i", lambda x: -x)
dslf.concrete("mod", f"(i, i) -> i", lambda x, y: x % y)
dslf.concrete("eq?", f"(i, i) -> b", lambda x, y: x == y)
dslf.concrete("gt?", f"(i, i) -> b", lambda x, y: x > y)
dslf.concrete(
    "is-prime",
    f"i -> b",
    lambda x: x > 1 and all(x % i for i in range(2, min(1 + int(x**0.5), x))),
)
dslf.concrete("is-square", f"i -> b", lambda x: x > 1 and int(x**0.5) ** 2 == x)
dslf.concrete("sum", f"[i] -> i", lambda x: sum(x))
# # (lambda (lambda (reduce (lambda (lambda (+ $0 $1))) 0 $0)))
for T in types:
    dslf.concrete(f"reverse_{T}", f"[{T}] -> [{T}]", lambda x: x[::-1])
    # # (lambda (reduce (lambda (lambda (++ (singleton $0) $1))) empty $0))
    dslf.concrete(
        f"all_{T}",
        f"((i) -> b, [{T}]) -> b",
        lambda f: lambda x: all(f(i) for i in x),
    )
    # # (lambda (lambda (reduce (lambda (lambda (and $0 $1))) true (map $1 $0))))
    dslf.concrete(
        f"any_{T}",
        f"((i) -> b, [{T}]) -> b",
        lambda f: lambda x: any(f(i) for i in x),
    )
    # # (lambda (lambda (reduce (lambda (lambda (or $0 $1))) true (map $1 $0))))
    dslf.concrete(f"index_{T}", f"(i, [{T}]) -> {T}", lambda x: lambda y: y[x])
    # # (lambda (lambda (reducei (lambda (lambda (lambda (if (eq? $1 $4) $0 0)))) 0 $0)))
    dslf.concrete(
        f"filter_{T}",
        f"({T} -> b, [{T}]) -> [{T}]",
        lambda f: lambda x: [i for i in x if f(i)],
    )
    # # (lambda (lambda (reduce (lambda (lambda (++ $1 (if ($3 $0) (singleton $0) empty)))) empty $0)))
    dslf.concrete(
        f"slice_{T}", f"(i, i, [{T}]) -> [{T}]", lambda x: lambda y: lambda z: z[x:y]
    )
    # # (lambda (lambda (lambda (reducei (lambda (lambda (lambda (++ $2 (if (and (or (gt? $1 $5) (eq? $1 $5)) (not (or (gt? $4 $1) (eq? $1 $4)))) (singleton $0) empty))))) empty $0))))

dsl = dslf.finalize()
