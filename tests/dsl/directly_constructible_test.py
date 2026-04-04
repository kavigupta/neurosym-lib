import unittest

import neurosym as ns
from neurosym.dsl.dsl_factory import _directly_constructible_types
from neurosym.types.type_signature import FunctionTypeSignature


def _t(*type_strs):
    return {ns.parse_type(t) for t in type_strs}


def _env(*type_strs):
    return frozenset(ns.parse_type(t) for t in type_strs)


class TestDirectlyConstructibleTypes(unittest.TestCase):
    def _sigs(self, *type_strs):
        return [FunctionTypeSignature.from_type(ns.parse_type(t)) for t in type_strs]

    def test_nullary_production(self):
        sigs = self._sigs("() -> i")
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_chain(self):
        sigs = self._sigs("() -> i", "i -> f")
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i", "f")})

    def test_unreachable_type(self):
        sigs = self._sigs("() -> i")
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_mutual_dependency(self):
        sigs = self._sigs("a -> b", "b -> a")
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): set()})

    def test_multi_arg_production(self):
        sigs = self._sigs("() -> i", "(i, i) -> i")
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_arrow_type_not_constructible_without_lambdas(self):
        sigs = self._sigs("() -> i")
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_arrow_type_constructible_with_lambdas(self):
        sigs = self._sigs("() -> i")
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_lambdas_unlock_production(self):
        sigs = self._sigs("() -> i", "(i -> i, i) -> i")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i")},
        )

    def test_nested_arrow_with_lambdas(self):
        sigs = self._sigs("() -> i", "(i -> i) -> f")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_type_variable_no_base(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("#x -> f")]
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): set()})

    def test_type_variable_with_base(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("#x -> f"), t.sig("() -> i")]
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i", "f")})

    def test_shared_type_variable(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("(#x, #x) -> #x"), t.sig("() -> i")]
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_shared_variable_produces_new_types(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("(#x) -> [#x]"), t.sig("() -> i")]
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=3)
        self.assertEqual(result, {frozenset(): _t("i", "[i]", "[[i]]", "[[[i]]]")})

    def test_shared_variable_with_lambdas(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("(#x -> #x) -> f"), t.sig("() -> i")]
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i", "f")})

    def test_depth_bound(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("(#x) -> [#x]"), t.sig("() -> i")]
        result = _directly_constructible_types(sigs, has_lambdas=False, max_depth=1)
        self.assertEqual(result, {frozenset(): _t("i", "[i]")})

    def test_call_needs_lambda(self):
        sigs = self._sigs("() -> i", "() -> f", "(i -> f, i) -> f")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "f")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_lambda_unlocks_new_output(self):
        sigs = self._sigs("() -> i", "() -> f", "(i -> f) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "f")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f", "g")},
        )

    def test_map_with_lambdas(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("() -> i"), t.sig("() -> [i]"), t.sig("(#a -> #b, [#a]) -> [#b]")]
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=2),
            {frozenset(): _t("i", "[i]")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=2),
            {frozenset(): _t("i", "[i]", "[[i]]")},
        )

    def test_map_produces_new_list_type(self):
        t = ns.TypeDefiner()
        sigs = [
            t.sig("() -> i"), t.sig("() -> f"), t.sig("() -> [i]"),
            t.sig("(#a -> #b, [#a]) -> [#b]"),
        ]
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=1),
            {frozenset(): _t("i", "f", "[i]")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=1),
            {frozenset(): _t("i", "f", "[i]", "[f]")},
        )

    def test_fold_with_lambdas(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("() -> i"), t.sig("() -> [i]"), t.sig("((#a, #a) -> #a, [#a]) -> #a")]
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "[i]")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "[i]")},
        )

    def test_compose_with_lambdas(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("() -> i"), t.sig("() -> f"), t.sig("(#a -> #b, #b -> #c) -> #a -> #c")]
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "f")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_lambda_chain_unlocks_deep(self):
        sigs = self._sigs("() -> i", "(i -> i) -> f", "(i -> f) -> g", "(i -> g) -> h")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f", "g", "h")},
        )

    def test_higher_order_needs_lambda_for_arg(self):
        sigs = self._sigs("() -> i", "() -> f", "() -> g", "((i -> f) -> g, i -> f) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "f", "g")},
        )
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f", "g")},
        )

    def test_lambda_with_type_var_in_arrow(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("() -> i"), t.sig("() -> f"), t.sig("(#x -> f) -> #x -> f")]
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_lambda_enables_list_of_functions(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("() -> i"), t.sig("(#x) -> [#x]")]
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=2),
            {frozenset(): _t("i", "[i]", "[[i]]")},
        )

    def test_arrow_type_directly_constructible(self):
        # () -> i -> f produces the arrow type i -> f directly, no lambdas needed.
        # (i -> f) -> g can then consume it.
        sigs = self._sigs("() -> i -> f", "(i -> f) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i -> f", "g")},
        )

    # --- Environment-aware tests ---

    def test_env_unlocks_production_needing_arrow(self):
        # (a, i) -> f: f constructible in env {a}.
        # (a -> f) -> g: a -> f is constructible (f constructible in env {a}), so g fires.
        sigs = self._sigs("() -> i", "(a, i) -> f", "(a -> f) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a"): _t("f")},
        )

    def test_env_chain(self):
        # a -> b, b -> c: in env {a}, a available, b producible, then c producible.
        # So a -> c constructible, and (a -> c) -> g fires.
        sigs = self._sigs("() -> i", "a -> b", "b -> c", "(a -> c) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a"): _t("b", "c")},
        )

    def test_env_multi_arg_arrow(self):
        # (a, b) -> f: in env {a, b}, both available, production fires.
        sigs = self._sigs("() -> i", "(a, b) -> f", "((a, b) -> f) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a", "b"): _t("f")},
        )

    def test_env_variable_directly_used(self):
        # a -> a: a is in env {a}, trivially constructible. (a -> a) -> g fires.
        sigs = self._sigs("() -> i", "(a -> a) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g")},
        )

    def test_env_not_needed_without_consumer(self):
        # (a, i) -> f: f constructible in env {a}, but no production consumes a -> f.
        # So env {a} is never explored.
        sigs = self._sigs("() -> i", "(a, i) -> f")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i")},
        )

    # --- Stress tests ---

    def test_env_chain_through_productions(self):
        # In env {a}: a -> b -> c -> d. (a -> d) -> g should fire.
        sigs = self._sigs("() -> i", "a -> b", "b -> c", "c -> d", "(a -> d) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a"): _t("b", "c", "d")},
        )

    def test_mutual_env_dependency(self):
        # (a) -> b and (b) -> a are mutually recursive.
        # In env {a}: b is producible. In env {b}: a is producible.
        # (a -> b) -> g and (b -> a) -> h should both fire.
        sigs = self._sigs("() -> i", "a -> b", "b -> a", "(a -> b) -> g", "(b -> a) -> h")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {
                frozenset(): _t("i", "g", "h"),
                _env("a"): _t("b"),
                _env("b"): _t("a"),
            },
        )

    def test_nested_arrow_env(self):
        # (a -> (b -> c)) -> g requires:
        # 1. env {a} created from outer arrow
        # 2. env {a, b} created from inner arrow
        # 3. (a, b) -> c fires in env {a, b}
        sigs = self._sigs("() -> i", "(a, b) -> c", "(a -> (b -> c)) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a", "b"): _t("c")},
        )

    def test_env_enables_type_var_binding(self):
        # In env {a}, production (a, i) -> f makes f available.
        # Signature (#x) -> [#x] in env {a} should bind #x to f and produce [f].
        # (a -> [f]) -> g should then fire.
        t = ns.TypeDefiner()
        sigs = [
            t.sig("() -> i"),
            t.sig("(a, i) -> f"),
            self._sigs("(a -> [f]) -> g")[0],
        ]
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        # (a -> [f]) -> g needs [f] constructible in env {a}. No production
        # creates [f], so g is NOT constructible.
        self.assertEqual(result, {
            frozenset(): _t("i"),
            _env("a"): _t("f"),
        })

    def test_env_production_output_is_arrow(self):
        # Production (a) -> i -> f: in env {a}, this directly produces the arrow type i -> f.
        # Then (i -> f) -> g should fire in env {a}.
        # And (a -> (i -> f)) -> h should fire at top level because i -> f is
        # constructible in env {a} (it's directly produced there).
        # But also (a -> g) -> k should fire since g is constructible in env {a}.
        sigs = self._sigs(
            "() -> i", "(a) -> i -> f", "(i -> f) -> g", "(a -> g) -> k",
        )
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {
            frozenset(): _t("i", "k"),
            _env("a"): _t("i -> f", "g"),
        })

    def test_lambda_and_env_interact(self):
        # (f -> f) -> g: f -> f is constructible even in empty env via lambda
        # (the lambda adds f to env, where f is trivially available).
        # So g is directly constructible.
        # (a -> g) -> h: g is directly constructible, so a -> g constructible via
        # lambda, so h is directly constructible.
        # (a, i) -> f: f is constructible in env {a}.
        sigs = self._sigs("() -> i", "(a, i) -> f", "(f -> f) -> g", "(a -> g) -> h")
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {
            frozenset(): _t("i", "g", "h"),
            _env("a"): _t("f"),
        })

    def test_two_arg_lambda_env(self):
        # (a, b) -> c is a production. Arrow type (a, b) -> c is constructible:
        # in env {a, b}, c is produced. So ((a, b) -> c) -> g fires.
        # Also test that single-arg arrows work: (a) -> c with (b -> c) in env...
        # Actually let's keep it simple.
        sigs = self._sigs("() -> i", "(a, b) -> c", "((a, b) -> c, i) -> g")
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {
            frozenset(): _t("i", "g"),
            _env("a", "b"): _t("c"),
        })

    def test_no_infinite_loop_on_self_referential(self):
        # (i -> i) -> i: with lambdas, i -> i is constructible (i is constructible).
        # So this just produces i (already constructible). Should terminate.
        sigs = self._sigs("() -> i", "(i -> i) -> i")
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_thunk_constructible(self):
        # () -> i is an arrow type with no inputs. Via lambda rule, it's constructible
        # if i is constructible (env doesn't grow). So (() -> i) -> f fires.
        sigs = self._sigs("() -> i", "(() -> i) -> f")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_deeply_nested_arrow_env(self):
        # a -> (b -> (c -> d)) requires envs {a}, {a,b}, {a,b,c} to be discovered.
        sigs = self._sigs("() -> i", "(a, b, c) -> d", "(a -> (b -> (c -> d))) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a", "b", "c"): _t("d")},
        )

    def test_transitive_env_production(self):
        # a -> b, (b, i) -> c: in env {a}, b produced, then c produced using b + i.
        sigs = self._sigs("() -> i", "a -> b", "(b, i) -> c", "(a -> c) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a"): _t("b", "c")},
        )

    def test_env_type_as_typevar_binding(self):
        # In env {a}, (a) -> b produces b. (#x) -> [#x] binds #x=b in env {a}
        # to produce [b]. Then (a -> [b]) -> g fires.
        t = ns.TypeDefiner()
        sigs = [
            t.sig("() -> i"), t.sig("a -> b"), t.sig("(#x) -> [#x]"),
            self._sigs("(a -> [b]) -> g")[0],
        ]
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=2)
        self.assertIn(ns.parse_type("g"), result[frozenset()])
        self.assertIn(ns.parse_type("[b]"), result[_env("a")])

    def test_production_output_same_as_env_member(self):
        # (a) -> a: in env {a}, produces a which is already in env.
        # Should be filtered as trivial. (a -> a) -> g still fires since
        # a is trivially constructible in env {a}.
        sigs = self._sigs("() -> i", "a -> a", "(a -> a) -> g")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g")},
        )

    # --- Bootstrap tests (types constructed from nothing via lambda) ---

    def test_bootstrap_identity_lambda(self):
        # (x -> x) -> x with no other productions. x -> x is constructible
        # via lambda (x is in env {x}), so x becomes directly constructible.
        sigs = self._sigs("(x -> x) -> x")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("x")},
        )
        # Without lambdas, x -> x is not constructible, so x is not constructible.
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): set()},
        )

    def test_bootstrap_cascades(self):
        # (x -> x) -> x bootstraps x. Then (x) -> y fires.
        sigs = self._sigs("(x -> x) -> x", "x -> y")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("x", "y")},
        )

    def test_bootstrap_multi_arg_arrow(self):
        # ((x, y) -> x) -> x: the arrow (x, y) -> x is constructible via lambda
        # because x is in env {x, y}. So x becomes directly constructible.
        sigs = self._sigs("((x, y) -> x) -> x")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("x")},
        )

    def test_bootstrap_nested(self):
        # (x -> y -> x) -> x: x -> (y -> x) is constructible via lambda because
        # y -> x is constructible in env {x} (x is in env {x, y}).
        sigs = self._sigs("(x -> y -> x) -> x")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("x")},
        )

    def test_bootstrap_does_not_help_wrong_type(self):
        # (x -> y) -> x: x -> y is constructible via lambda only if y is
        # constructible in env {x}. y is not in env {x} and no production
        # makes it. So x is NOT constructible.
        sigs = self._sigs("(x -> y) -> x")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): set()},
        )

    # --- Real DSL tests ---

    def test_basic_arith_no_lambdas(self):
        sigs = self._sigs("(i, i) -> i", "() -> i")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i")},
        )

    def test_basic_arith_with_lambdas(self):
        # i is directly constructible. Arrow types like i -> i are constructible
        # but not directly constructible (no production outputs them).
        sigs = self._sigs("(i, i) -> i", "() -> i")
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i")},
        )

    def test_dreamcoder_dsl(self):
        sigs = self._sigs(
            "() -> f", "() -> f", "() -> f",      # constants
            "(f, f) -> f", "(f, f) -> f",          # arith ops
            "(f, f) -> f", "(f, f) -> f",
            "(f, f) -> f",
            "f -> f", "f -> f",                    # sin, sqrt
            "(f, f) -> b",                         # <
            "(b, f, f) -> f",                      # ite
        )
        result = _directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        direct = result[frozenset()]
        # f and b are the only non-arrow types that should be directly constructible
        self.assertEqual(direct, _t("f", "b"))

    def test_bouncing_ball_dsl(self):
        # Nullary prods output {f,4}->{f,1} and {f,4}->{f,4}.
        # ite with #a={f,4} reuses those. map lifts to list levels up to depth 5.
        t = ns.TypeDefiner(L=4)
        t.typedef("fL", "{f, $L}")
        sigs = [
            t.sig("() -> $fL -> {f, 1}"),
            t.sig("() -> $fL -> $fL"),
            t.sig("(#a -> {f, 1}, #a -> #a, #a -> #a) -> #a -> #a"),
            t.sig("(#a -> #b) -> [#a] -> [#b]"),
        ]
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=5),
            {frozenset(): _t(
                "{f, 4} -> {f, 1}", "{f, 4} -> {f, 4}",
                "[{f, 4}] -> [{f, 1}]", "[{f, 4}] -> [{f, 4}]",
                "[[{f, 4}]] -> [[{f, 1}]]", "[[{f, 4}]] -> [[{f, 4}]]",
                "[[[{f, 4}]]] -> [[[{f, 1}]]]", "[[[{f, 4}]]] -> [[[{f, 4}]]]",
            )},
        )

    def test_rnn_dsl(self):
        # Nullary: ({f,12},{f,12})->{f,12}, {f,12}->f, {f,12}->{f,12}.
        # fold, map, compose, output chain these through list levels.
        t = ns.TypeDefiner(L=12, O=4)
        t.typedef("fL", "{f, $L}")
        sigs = [
            t.sig("() -> ($fL, $fL) -> $fL"),
            t.sig("() -> ($fL, $fL) -> $fL"),
            t.sig("((#a, #a) -> #a) -> [#a] -> #a"),
            t.sig("() -> $fL -> f"),
            t.sig("() -> $fL -> $fL"),
            t.sig("(([$fL]) -> [$fL]) -> [$fL] -> [{f, $O}]"),
            t.sig("(#a -> [f], #a -> #a, #a -> #a) -> #a -> #a"),
            t.sig("(#a -> #b) -> [#a] -> [#b]"),
            t.sig("(#a -> #b, #b -> #c) -> #a -> #c"),
        ]
        self.assertEqual(
            _directly_constructible_types(sigs, has_lambdas=False, max_depth=5),
            {frozenset(): _t(
                # nullary
                "({f, 12}, {f, 12}) -> {f, 12}",
                "{f, 12} -> f",
                "{f, 12} -> {f, 12}",
                # fold: [{f,12}] -> {f,12}
                # map: [{f,12}] -> [{f,12}], [{f,12}] -> [f]
                # compose: {f,12} -> f (already), [{f,12}] -> f, etc.
                # output: [{f,12}] -> [{f,4}]
                "[{f, 12}] -> f",
                "[{f, 12}] -> {f, 12}",
                "[{f, 12}] -> [f]",
                "[{f, 12}] -> [{f, 12}]",
                "[{f, 12}] -> [{f, 4}]",
                # depth 2 list nesting
                "[[{f, 12}]] -> f",
                "[[{f, 12}]] -> {f, 12}",
                "[[{f, 12}]] -> [f]",
                "[[{f, 12}]] -> [{f, 12}]",
                "[[{f, 12}]] -> [{f, 4}]",
                "[[{f, 12}]] -> [[f]]",
                "[[{f, 12}]] -> [[{f, 12}]]",
                "[[{f, 12}]] -> [[{f, 4}]]",
                # depth 3 list nesting
                "[[[{f, 12}]]] -> f",
                "[[[{f, 12}]]] -> {f, 12}",
                "[[[{f, 12}]]] -> [f]",
                "[[[{f, 12}]]] -> [{f, 12}]",
                "[[[{f, 12}]]] -> [{f, 4}]",
                "[[[{f, 12}]]] -> [[f]]",
                "[[[{f, 12}]]] -> [[{f, 12}]]",
                "[[[{f, 12}]]] -> [[{f, 4}]]",
                "[[[{f, 12}]]] -> [[[f]]]",
                "[[[{f, 12}]]] -> [[[{f, 12}]]]",
                "[[[{f, 12}]]] -> [[[{f, 4}]]]",
            )},
        )
