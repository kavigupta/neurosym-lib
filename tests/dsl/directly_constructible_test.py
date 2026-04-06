import unittest

import neurosym as ns


def _t(*type_strs):
    return {ns.parse_type(t) for t in type_strs}


def _env(*type_strs):
    return frozenset(ns.parse_type(t) for t in type_strs)


class TestDirectlyConstructibleTypes(unittest.TestCase):
    def _sigs(self, *type_strs):
        return [ns.FunctionTypeSignature.from_type(ns.parse_type(t)) for t in type_strs]

    def test_nullary_production(self):
        sigs = self._sigs("() -> i")
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_chain(self):
        sigs = self._sigs("() -> i", "i -> f")
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i", "f")})

    def test_unreachable_type(self):
        sigs = self._sigs("() -> i")
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_mutual_dependency(self):
        sigs = self._sigs("a -> b", "b -> a")
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): set()})

    def test_multi_arg_production(self):
        sigs = self._sigs("() -> i", "(i, i) -> i")
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_arrow_type_not_constructible_without_lambdas(self):
        sigs = self._sigs("() -> i")
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_arrow_type_constructible_with_lambdas(self):
        sigs = self._sigs("() -> i")
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_lambdas_unlock_production(self):
        sigs = self._sigs("() -> i", "(i -> i, i) -> i")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i")},
        )

    def test_nested_arrow_with_lambdas(self):
        sigs = self._sigs("() -> i", "(i -> i) -> f")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_type_variable_no_base(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("#x -> f")]
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): set()})

    def test_type_variable_with_base(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("#x -> f"), t.sig("() -> i")]
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i", "f")})

    def test_shared_type_variable(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("(#x, #x) -> #x"), t.sig("() -> i")]
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_shared_variable_produces_new_types(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("(#x) -> [#x]"), t.sig("() -> i")]
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=3)
        self.assertEqual(result, {frozenset(): _t("i", "[i]", "[[i]]", "[[[i]]]")})

    def test_shared_variable_with_lambdas(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("(#x -> #x) -> f"), t.sig("() -> i")]
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i", "f")})

    def test_depth_bound(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("(#x) -> [#x]"), t.sig("() -> i")]
        result = ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=1)
        self.assertEqual(result, {frozenset(): _t("i", "[i]")})

    def test_call_needs_lambda(self):
        sigs = self._sigs("() -> i", "() -> f", "(i -> f, i) -> f")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "f")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_lambda_unlocks_new_output(self):
        sigs = self._sigs("() -> i", "() -> f", "(i -> f) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "f")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f", "g")},
        )

    def test_map_with_lambdas(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("() -> i"), t.sig("() -> [i]"), t.sig("(#a -> #b, [#a]) -> [#b]")]
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=2),
            {frozenset(): _t("i", "[i]")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=2),
            {frozenset(): _t("i", "[i]", "[[i]]")},
        )

    def test_map_produces_new_list_type(self):
        t = ns.TypeDefiner()
        sigs = [
            t.sig("() -> i"),
            t.sig("() -> f"),
            t.sig("() -> [i]"),
            t.sig("(#a -> #b, [#a]) -> [#b]"),
        ]
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=1),
            {frozenset(): _t("i", "f", "[i]")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=1),
            {frozenset(): _t("i", "f", "[i]", "[f]")},
        )

    def test_fold_with_lambdas(self):
        t = ns.TypeDefiner()
        sigs = [
            t.sig("() -> i"),
            t.sig("() -> [i]"),
            t.sig("((#a, #a) -> #a, [#a]) -> #a"),
        ]
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "[i]")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "[i]")},
        )

    def test_compose_with_lambdas(self):
        t = ns.TypeDefiner()
        sigs = [
            t.sig("() -> i"),
            t.sig("() -> f"),
            t.sig("(#a -> #b, #b -> #c) -> #a -> #c"),
        ]
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "f")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_lambda_chain_unlocks_deep(self):
        sigs = self._sigs("() -> i", "(i -> i) -> f", "(i -> f) -> g", "(i -> g) -> h")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f", "g", "h")},
        )

    def test_higher_order_needs_lambda_for_arg(self):
        sigs = self._sigs(
            "() -> i", "() -> f", "() -> g", "((i -> f) -> g, i -> f) -> g"
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i", "f", "g")},
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f", "g")},
        )

    def test_lambda_with_type_var_in_arrow(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("() -> i"), t.sig("() -> f"), t.sig("(#x -> f) -> #x -> f")]
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_lambda_enables_list_of_functions(self):
        t = ns.TypeDefiner()
        sigs = [t.sig("() -> i"), t.sig("(#x) -> [#x]")]
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=2),
            {frozenset(): _t("i", "[i]", "[[i]]")},
        )

    def test_arrow_type_directly_constructible(self):
        # () -> i -> f produces the arrow type i -> f directly, no lambdas needed.
        # (i -> f) -> g can then consume it.
        sigs = self._sigs("() -> i -> f", "(i -> f) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i -> f", "g")},
        )

    # --- Environment-aware tests ---

    def test_env_unlocks_production_needing_arrow(self):
        # (a, i) -> f: f constructible in env {a}.
        # (a -> f) -> g: a -> f is constructible (f constructible in env {a}), so g fires.
        sigs = self._sigs("() -> i", "(a, i) -> f", "(a -> f) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a"): _t("f")},
        )

    def test_env_chain(self):
        # a -> b, b -> c: in env {a}, a available, b producible, then c producible.
        # So a -> c constructible, and (a -> c) -> g fires.
        sigs = self._sigs("() -> i", "a -> b", "b -> c", "(a -> c) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a"): _t("b", "c")},
        )

    def test_env_multi_arg_arrow(self):
        # (a, b) -> f: in env {a, b}, both available, production fires.
        sigs = self._sigs("() -> i", "(a, b) -> f", "((a, b) -> f) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a", "b"): _t("f")},
        )

    def test_env_variable_directly_used(self):
        # a -> a: a is in env {a}, trivially constructible. (a -> a) -> g fires.
        sigs = self._sigs("() -> i", "(a -> a) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g")},
        )

    def test_env_not_needed_without_consumer(self):
        # (a, i) -> f: f constructible in env {a}, but no production consumes a -> f.
        # So env {a} is never explored.
        sigs = self._sigs("() -> i", "(a, i) -> f")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i")},
        )

    # --- Stress tests ---

    def test_env_chain_through_productions(self):
        # In env {a}: a -> b -> c -> d. (a -> d) -> g should fire.
        sigs = self._sigs("() -> i", "a -> b", "b -> c", "c -> d", "(a -> d) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a"): _t("b", "c", "d")},
        )

    def test_mutual_env_dependency(self):
        # (a) -> b and (b) -> a are mutually recursive.
        # In env {a}: b is producible. In env {b}: a is producible.
        # (a -> b) -> g and (b -> a) -> h should both fire.
        sigs = self._sigs(
            "() -> i", "a -> b", "b -> a", "(a -> b) -> g", "(b -> a) -> h"
        )
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
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
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
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
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        # (a -> [f]) -> g needs [f] constructible in env {a}. No production
        # creates [f], so g is NOT constructible.
        self.assertEqual(
            result,
            {
                frozenset(): _t("i"),
                _env("a"): _t("f"),
            },
        )

    def test_env_production_output_is_arrow(self):
        # Production (a) -> i -> f: in env {a}, this directly produces the arrow type i -> f.
        # Then (i -> f) -> g should fire in env {a}.
        # And (a -> (i -> f)) -> h should fire at top level because i -> f is
        # constructible in env {a} (it's directly produced there).
        # But also (a -> g) -> k should fire since g is constructible in env {a}.
        sigs = self._sigs(
            "() -> i",
            "(a) -> i -> f",
            "(i -> f) -> g",
            "(a -> g) -> k",
        )
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(
            result,
            {
                frozenset(): _t("i", "k"),
                _env("a"): _t("i -> f", "g"),
            },
        )

    def test_lambda_and_env_interact(self):
        # (f -> f) -> g: f -> f is constructible even in empty env via lambda
        # (the lambda adds f to env, where f is trivially available).
        # So g is directly constructible.
        # (a -> g) -> h: g is directly constructible, so a -> g constructible via
        # lambda, so h is directly constructible.
        # (a, i) -> f: f is constructible in env {a}.
        sigs = self._sigs("() -> i", "(a, i) -> f", "(f -> f) -> g", "(a -> g) -> h")
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(
            result,
            {
                frozenset(): _t("i", "g", "h"),
                _env("a"): _t("f"),
            },
        )

    def test_two_arg_lambda_env(self):
        # (a, b) -> c is a production. Arrow type (a, b) -> c is constructible:
        # in env {a, b}, c is produced. So ((a, b) -> c) -> g fires.
        # Also test that single-arg arrows work: (a) -> c with (b -> c) in env...
        # Actually let's keep it simple.
        sigs = self._sigs("() -> i", "(a, b) -> c", "((a, b) -> c, i) -> g")
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(
            result,
            {
                frozenset(): _t("i", "g"),
                _env("a", "b"): _t("c"),
            },
        )

    def test_no_infinite_loop_on_self_referential(self):
        # (i -> i) -> i: with lambdas, i -> i is constructible (i is constructible).
        # So this just produces i (already constructible). Should terminate.
        sigs = self._sigs("() -> i", "(i -> i) -> i")
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
        self.assertEqual(result, {frozenset(): _t("i")})

    def test_thunk_constructible(self):
        # () -> i is an arrow type with no inputs. Via lambda rule, it's constructible
        # if i is constructible (env doesn't grow). So (() -> i) -> f fires.
        sigs = self._sigs("() -> i", "(() -> i) -> f")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "f")},
        )

    def test_deeply_nested_arrow_env(self):
        # a -> (b -> (c -> d)) requires envs {a}, {a,b}, {a,b,c} to be discovered.
        sigs = self._sigs("() -> i", "(a, b, c) -> d", "(a -> (b -> (c -> d))) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a", "b", "c"): _t("d")},
        )

    def test_transitive_env_production(self):
        # a -> b, (b, i) -> c: in env {a}, b produced, then c produced using b + i.
        sigs = self._sigs("() -> i", "a -> b", "(b, i) -> c", "(a -> c) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g"), _env("a"): _t("b", "c")},
        )

    def test_env_type_as_typevar_binding(self):
        # In env {a}, (a) -> b produces b. (#x) -> [#x] binds #x=b in env {a}
        # to produce [b]. Then (a -> [b]) -> g fires.
        t = ns.TypeDefiner()
        sigs = [
            t.sig("() -> i"),
            t.sig("a -> b"),
            t.sig("(#x) -> [#x]"),
            self._sigs("(a -> [b]) -> g")[0],
        ]
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=2)
        self.assertIn(ns.parse_type("g"), result[frozenset()])
        self.assertIn(ns.parse_type("[b]"), result[_env("a")])

    def test_production_output_same_as_env_member(self):
        # (a) -> a: in env {a}, produces a which is already in env.
        # Should be filtered as trivial. (a -> a) -> g still fires since
        # a is trivially constructible in env {a}.
        sigs = self._sigs("() -> i", "a -> a", "(a -> a) -> g")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i", "g")},
        )

    def test_no_nullary_productions_with_target_env(self):
        # No nullary productions: {f,1} -> {f,2}, {f,2} -> {f,3}, {f,3} -> {f,4}.
        # Nothing is directly constructible in empty env.
        # But with target {f,1} -> {f,4}, env {{f,1}} is seeded and the chain fires.
        sigs = self._sigs("{f, 1} -> {f, 2}", "{f, 2} -> {f, 3}", "{f, 3} -> {f, 4}")
        # Without target_types, nothing is constructible
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): set()},
        )
        # With target_types, env {{f,1}} is seeded and the chain fires
        self.assertEqual(
            ns.directly_constructible_types(
                sigs,
                has_lambdas=True,
                max_depth=6,
                target_types=[ns.parse_type("{f, 1} -> {f, 4}")],
            ),
            {
                frozenset(): set(),
                _env("{f, 1}"): _t("{f, 2}", "{f, 3}", "{f, 4}"),
            },
        )

    def test_no_nullary_reachable_with_target_env(self):
        # Same as above but checking reachable_symbols finds all productions.
        t = ns.TypeDefiner()
        named_sigs = [
            ("step1", t.sig("{f, 1} -> {f, 2}")),
            ("step2", t.sig("{f, 2} -> {f, 3}")),
            ("step3", t.sig("{f, 3} -> {f, 4}")),
        ]
        sigs_only = [s for _, s in named_sigs]
        target = ns.parse_type("{f, 1} -> {f, 4}")
        ct = ns.directly_constructible_types(
            sigs_only,
            has_lambdas=True,
            max_depth=6,
            target_types=[target],
        )
        prods, lams = ns.reachable_symbols(
            named_sigs,
            ct,
            [target],
            has_lambdas=True,
            max_depth=6,
            max_lambda_depth=6,
        )
        self.assertEqual(
            _render_prods(prods),
            {
                ("step1", "{f, 1} -> {f, 2}"),
                ("step2", "{f, 2} -> {f, 3}"),
                ("step3", "{f, 3} -> {f, 4}"),
            },
        )
        self.assertEqual(lams, {(ns.parse_type("{f, 1}"),)})

    # --- Bootstrap tests (types constructed from nothing via lambda) ---

    def test_bootstrap_identity_lambda(self):
        # (x -> x) -> x with no other productions. x -> x is constructible
        # via lambda (x is in env {x}), so x becomes directly constructible.
        sigs = self._sigs("(x -> x) -> x")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("x")},
        )
        # Without lambdas, x -> x is not constructible, so x is not constructible.
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): set()},
        )

    def test_bootstrap_cascades(self):
        # (x -> x) -> x bootstraps x. Then (x) -> y fires.
        sigs = self._sigs("(x -> x) -> x", "x -> y")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("x", "y")},
        )

    def test_bootstrap_multi_arg_arrow(self):
        # ((x, y) -> x) -> x: the arrow (x, y) -> x is constructible via lambda
        # because x is in env {x, y}. So x becomes directly constructible.
        sigs = self._sigs("((x, y) -> x) -> x")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("x")},
        )

    def test_bootstrap_nested(self):
        # (x -> y -> x) -> x: x -> (y -> x) is constructible via lambda because
        # y -> x is constructible in env {x} (x is in env {x, y}).
        sigs = self._sigs("(x -> y -> x) -> x")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("x")},
        )

    def test_bootstrap_does_not_help_wrong_type(self):
        # (x -> y) -> x: x -> y is constructible via lambda only if y is
        # constructible in env {x}. y is not in env {x} and no production
        # makes it. So x is NOT constructible.
        sigs = self._sigs("(x -> y) -> x")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): set()},
        )

    # --- Real DSL tests ---

    def test_basic_arith_no_lambdas(self):
        sigs = self._sigs("(i, i) -> i", "() -> i")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=6),
            {frozenset(): _t("i")},
        )

    def test_basic_arith_with_lambdas(self):
        # i is directly constructible. Arrow types like i -> i are constructible
        # but not directly constructible (no production outputs them).
        sigs = self._sigs("(i, i) -> i", "() -> i")
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6),
            {frozenset(): _t("i")},
        )

    def test_dreamcoder_dsl(self):
        sigs = self._sigs(
            "() -> f",
            "() -> f",
            "() -> f",  # constants
            "(f, f) -> f",
            "(f, f) -> f",  # arith ops
            "(f, f) -> f",
            "(f, f) -> f",
            "(f, f) -> f",
            "f -> f",
            "f -> f",  # sin, sqrt
            "(f, f) -> b",  # <
            "(b, f, f) -> f",  # ite
        )
        result = ns.directly_constructible_types(sigs, has_lambdas=True, max_depth=6)
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
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=5),
            {
                frozenset(): _t(
                    "{f, 4} -> {f, 1}",
                    "{f, 4} -> {f, 4}",
                    "[{f, 4}] -> [{f, 1}]",
                    "[{f, 4}] -> [{f, 4}]",
                    "[[{f, 4}]] -> [[{f, 1}]]",
                    "[[{f, 4}]] -> [[{f, 4}]]",
                    "[[[{f, 4}]]] -> [[[{f, 1}]]]",
                    "[[[{f, 4}]]] -> [[[{f, 4}]]]",
                )
            },
        )

    def test_rnn_dsl(self):
        # RNN DSL with scan (not fold). Nullary prods, scan, map, compose,
        # output, ite chain through list levels. scan keeps output as list
        # (unlike fold which reduced to scalar).
        t = ns.TypeDefiner(L=12, O=4)
        t.typedef("fL", "{f, $L}")
        sigs = [
            t.sig("() -> ($fL, $fL) -> $fL"),
            t.sig("() -> ($fL, $fL) -> $fL"),
            t.sig("((#a, #a) -> #a) -> [#a] -> [#a]"),  # scan
            t.sig("() -> $fL -> f"),
            t.sig("() -> $fL -> $fL"),
            t.sig("(([$fL]) -> [$fL]) -> [$fL] -> [{f, $O}]"),
            t.sig("(#a -> [f], #a -> #a, #a -> #a) -> #a -> #a"),
            t.sig("(#a -> #b) -> [#a] -> [#b]"),
            t.sig("(#a -> #b, #b -> #c) -> #a -> #c"),
        ]
        self.assertEqual(
            ns.directly_constructible_types(sigs, has_lambdas=False, max_depth=5),
            {
                frozenset(): _t(
                    "({f, 12}, {f, 12}) -> {f, 12}",
                    "{f, 12} -> f",
                    "{f, 12} -> {f, 12}",
                    "[{f, 12}] -> [f]",
                    "[{f, 12}] -> [{f, 12}]",
                    "[{f, 12}] -> [{f, 4}]",
                    "[[{f, 12}]] -> [[f]]",
                    "[[{f, 12}]] -> [[{f, 12}]]",
                    "[[{f, 12}]] -> [[{f, 4}]]",
                    "[[[{f, 12}]]] -> [[[f]]]",
                    "[[[{f, 12}]]] -> [[[{f, 12}]]]",
                    "[[[{f, 12}]]] -> [[[{f, 4}]]]",
                )
            },
        )


def _render_prods(prod_sigs):
    """Convert {sym: [sig, ...]} to {(sym, type_str), ...} for easy comparison."""
    return {(sym, sig.render()) for sym, sigs in prod_sigs.items() for sig in sigs}


class TestReachableSymbols(unittest.TestCase):
    def _named_sigs(self, *named_type_strs):
        """Build (symbol, sig) pairs from 'name :: type_str' strings."""
        result = []
        for s in named_type_strs:
            name, type_str = s.split(" :: ", 1)
            result.append(
                (name, ns.FunctionTypeSignature.from_type(ns.parse_type(type_str)))
            )
        return result

    def _run(self, named_sigs, targets, has_lambdas=False, max_depth=6):
        sigs_only = [s for _, s in named_sigs]
        ct = ns.directly_constructible_types(
            sigs_only,
            has_lambdas=has_lambdas,
            max_depth=max_depth,
        )
        return ns.reachable_symbols(
            named_sigs,
            ct,
            [ns.parse_type(t) for t in targets],
            has_lambdas=has_lambdas,
            max_depth=max_depth,
            max_lambda_depth=max_depth,
        )

    def test_basic(self):
        sigs = self._named_sigs("one :: () -> i", "add :: (i, i) -> i")
        prods, lams = self._run(sigs, ["i"])
        self.assertEqual(
            _render_prods(prods),
            {("one", "() -> i"), ("add", "(i, i) -> i")},
        )
        self.assertEqual(lams, set())

    def test_unreachable_production(self):
        sigs = self._named_sigs("one :: () -> i", "to_f :: i -> f")
        prods, lams = self._run(sigs, ["i"])
        self.assertEqual(_render_prods(prods), {("one", "() -> i")})
        self.assertEqual(lams, set())

    def test_chain(self):
        sigs = self._named_sigs("one :: () -> i", "to_f :: i -> f")
        prods, lams = self._run(sigs, ["f"])
        self.assertEqual(
            _render_prods(prods),
            {("one", "() -> i"), ("to_f", "i -> f")},
        )
        self.assertEqual(lams, set())

    def test_lambda_reaches_base(self):
        sigs = self._named_sigs("one :: () -> i", "use_fn :: (i -> i) -> f")
        prods, lams = self._run(sigs, ["f"], has_lambdas=True)
        self.assertEqual(
            _render_prods(prods),
            {("one", "() -> i"), ("use_fn", "(i -> i) -> f")},
        )
        self.assertEqual(lams, {(ns.parse_type("i"),)})

    def test_lambda_not_available_without_flag(self):
        sigs = self._named_sigs("one :: () -> i", "use_fn :: (i -> i) -> f")
        prods, lams = self._run(sigs, ["f"], has_lambdas=False)
        self.assertEqual(_render_prods(prods), set())
        self.assertEqual(lams, set())

    def test_type_variable_substs(self):
        t = ns.TypeDefiner()
        sigs = [("one", t.sig("() -> i")), ("convert", t.sig("#x -> f"))]
        prods, lams = self._run(sigs, ["f"])
        self.assertEqual(
            _render_prods(prods),
            {("one", "() -> i"), ("convert", "i -> f"), ("convert", "f -> f")},
        )
        self.assertEqual(lams, set())

    def test_multiple_bases_reachable(self):
        t = ns.TypeDefiner()
        sigs = [
            ("one_i", t.sig("() -> i")),
            ("one_f", t.sig("() -> f")),
            ("convert", t.sig("#x -> g")),
        ]
        prods, lams = self._run(sigs, ["g"])
        self.assertEqual(
            _render_prods(prods),
            {
                ("one_i", "() -> i"),
                ("one_f", "() -> f"),
                ("convert", "i -> g"),
                ("convert", "f -> g"),
                ("convert", "g -> g"),
            },
        )
        self.assertEqual(lams, set())

    def test_unconstructible_input(self):
        sigs = self._named_sigs("one :: () -> i", "needs_f :: (f, i) -> g")
        prods, lams = self._run(sigs, ["g"])
        self.assertEqual(_render_prods(prods), set())
        self.assertEqual(lams, set())

    def test_env_aware_reachability(self):
        sigs = self._named_sigs(
            "one :: () -> i",
            "make_f :: (a, i) -> f",
            "use_fn :: (a -> f) -> g",
        )
        prods, lams = self._run(sigs, ["g"], has_lambdas=True)
        self.assertEqual(
            _render_prods(prods),
            {
                ("one", "() -> i"),
                ("make_f", "(a, i) -> f"),
                ("use_fn", "(a -> f) -> g"),
            },
        )
        self.assertEqual(lams, {(ns.parse_type("a"),)})

    def test_bootstrap_reachable(self):
        sigs = self._named_sigs("boot :: (x -> x) -> x", "to_f :: x -> f")
        prods, lams = self._run(sigs, ["f"], has_lambdas=True)
        self.assertEqual(
            _render_prods(prods),
            {("boot", "(x -> x) -> x"), ("to_f", "x -> f")},
        )
        self.assertEqual(lams, {(ns.parse_type("x"),)})

    def test_multiple_targets(self):
        sigs = self._named_sigs("one :: () -> i", "to_f :: i -> f", "to_g :: f -> g")
        prods, lams = self._run(sigs, ["i", "g"])
        self.assertEqual(
            _render_prods(prods),
            {("one", "() -> i"), ("to_f", "i -> f"), ("to_g", "f -> g")},
        )
        self.assertEqual(lams, set())

    def test_nested_lambda_types(self):
        sigs = self._named_sigs(
            "one :: () -> i",
            "one_f :: () -> f",
            "one_g :: () -> g",
            "use :: (i -> (f -> g)) -> h",
        )
        prods, lams = self._run(sigs, ["h"], has_lambdas=True)
        self.assertEqual(
            _render_prods(prods),
            {("one_g", "() -> g"), ("use", "(i -> f -> g) -> h")},
        )
        self.assertEqual(lams, {(ns.parse_type("i"),), (ns.parse_type("f"),)})

    def test_rnn_dsl_target(self):
        # RNN DSL with scan (not fold) so all productions are reachable.
        t = ns.TypeDefiner(L=12, O=4)
        t.typedef("fL", "{f, $L}")
        sigs = [
            ("add", t.sig("() -> ($fL, $fL) -> $fL")),
            ("mul", t.sig("() -> ($fL, $fL) -> $fL")),
            ("scan", t.sig("((#a, #a) -> #a) -> [#a] -> [#a]")),
            ("sum", t.sig("() -> $fL -> f")),
            ("linear", t.sig("() -> $fL -> $fL")),
            ("output", t.sig("(([$fL]) -> [$fL]) -> [$fL] -> [{f, $O}]")),
            ("ite", t.sig("(#a -> [f], #a -> #a, #a -> #a) -> #a -> #a")),
            ("map", t.sig("(#a -> #b) -> [#a] -> [#b]")),
            ("compose", t.sig("(#a -> #b, #b -> #c) -> #a -> #c")),
        ]
        sigs_only = [s for _, s in sigs]
        ct = ns.directly_constructible_types(sigs_only, has_lambdas=False, max_depth=5)
        prods, lams = ns.reachable_symbols(
            sigs,
            ct,
            [ns.parse_type("[{f, 12}] -> [{f, 4}]")],
            has_lambdas=False,
            max_depth=5,
            max_lambda_depth=5,
        )
        self.assertEqual(
            _render_prods(prods),
            {
                ("add", "() -> ({f, 12}, {f, 12}) -> {f, 12}"),
                ("mul", "() -> ({f, 12}, {f, 12}) -> {f, 12}"),
                ("scan", "((#a, #a) -> #a) -> [#a] -> [#a]"),
                # compose: #b is non-shared, so it gets substituted;
                # #a and #c are shared and preserved
                ("compose", "(#a -> [{f, 12}], [{f, 12}] -> #c) -> #a -> #c"),
                ("compose", "(#a -> {f, 12}, {f, 12} -> #c) -> #a -> #c"),
                ("ite", "(#a -> [f], #a -> #a, #a -> #a) -> #a -> #a"),
                ("linear", "() -> {f, 12} -> {f, 12}"),
                ("map", "(#a -> #b) -> [#a] -> [#b]"),
                ("output", "([{f, 12}] -> [{f, 12}]) -> [{f, 12}] -> [{f, 4}]"),
                ("sum", "() -> {f, 12} -> f"),
            },
        )
        self.assertEqual(lams, set())

    def test_type_variable_combinatorial(self):
        # List DSL with type variables, lambdas, and multi-arg higher-order
        # productions. This tests that reachable_symbols doesn't hang from
        # combinatorial explosion of type variable bindings.
        t = ns.TypeDefiner()
        sigs = [
            ("zero", t.sig("() -> i")),
            ("true", t.sig("() -> b")),
            ("empty", t.sig("() -> [#T]")),
            ("singleton", t.sig("#T -> [#T]")),
            ("range", t.sig("i -> [i]")),
            ("concat", t.sig("([#T], [#T]) -> [#T]")),
            ("mapi", t.sig("((i, #T) -> #R, [#T]) -> [#R]")),
            ("reducei", t.sig("((i, #R, #T) -> #R, #R, [#T]) -> #R")),
            ("sort", t.sig("[#T] -> [#T]")),
            ("reverse", t.sig("[#T] -> [#T]")),
            ("sum", t.sig("[i] -> i")),
            ("index", t.sig("(i, [#T]) -> #T")),
            ("filter", t.sig("(#T -> b, [#T]) -> [#T]")),
            ("all", t.sig("((#T) -> b, [#T]) -> b")),
            ("any", t.sig("((#T) -> b, [#T]) -> b")),
            ("slice", t.sig("(i, i, [#T]) -> [#T]")),
            ("ite", t.sig("(b, #T, #T) -> #T")),
            ("not", t.sig("b -> b")),
            ("and", t.sig("(b, b) -> b")),
            ("or", t.sig("(b, b) -> b")),
            ("eq", t.sig("(i, i) -> b")),
            ("gt", t.sig("(i, i) -> b")),
            ("add", t.sig("(i, i) -> i")),
            ("mul", t.sig("(i, i) -> i")),
            ("negate", t.sig("i -> i")),
            ("mod", t.sig("(i, i) -> i")),
        ]
        sigs_only = [s for _, s in sigs]
        ct = ns.directly_constructible_types(
            sigs_only,
            has_lambdas=True,
            max_depth=5,
            target_types=[ns.parse_type("[i] -> i")],
        )
        prods, _ = ns.reachable_symbols(
            sigs,
            ct,
            [ns.parse_type("[i] -> i")],
            has_lambdas=True,
            max_depth=5,
            max_lambda_depth=5,
        )
        # All productions should be reachable
        self.assertEqual(set(prods.keys()), {s for s, _ in sigs})

    def test_basic_arith_with_lambdas(self):
        # Targets include arrow types: i -> i needs lambda (i,),
        # (i, i) -> i needs lambda (i, i).
        sigs = self._named_sigs("plus :: (i, i) -> i", "one :: () -> i")
        prods, lams = self._run(sigs, ["i", "i -> i", "(i, i) -> i"], has_lambdas=True)
        self.assertEqual(
            _render_prods(prods),
            {("one", "() -> i"), ("plus", "(i, i) -> i")},
        )
        self.assertEqual(
            lams,
            {(ns.parse_type("i"),), (ns.parse_type("i"), ns.parse_type("i"))},
        )

    def test_dreamcoder_with_lambdas(self):
        # Symbolic regression DSL. Target: f -> f (lambdas produce the function).
        # All productions are reachable since they all produce f or b
        # (and b is needed by ite).
        sigs = self._named_sigs(
            "zero :: () -> f",
            "one :: () -> f",
            "two :: () -> f",
            "plus :: (f, f) -> f",
            "minus :: (f, f) -> f",
            "times :: (f, f) -> f",
            "power :: (f, f) -> f",
            "divide :: (f, f) -> f",
            "sin :: f -> f",
            "sqrt :: f -> f",
            "lt :: (f, f) -> b",
            "ite :: (b, f, f) -> f",
        )
        prods, lams = self._run(sigs, ["f -> f"], has_lambdas=True)
        self.assertEqual(
            _render_prods(prods),
            {
                ("zero", "() -> f"),
                ("one", "() -> f"),
                ("two", "() -> f"),
                ("plus", "(f, f) -> f"),
                ("minus", "(f, f) -> f"),
                ("times", "(f, f) -> f"),
                ("power", "(f, f) -> f"),
                ("divide", "(f, f) -> f"),
                ("sin", "f -> f"),
                ("sqrt", "f -> f"),
                ("lt", "(f, f) -> b"),
                ("ite", "(b, f, f) -> f"),
            },
        )
        self.assertEqual(lams, {(ns.parse_type("f"),)})
