import sys
import unittest

from parameterized import parameterized

import neurosym as ns

from .utils import cwq, fit_to, small_set_runnable_code_examples


class DefUseMaskTestGeneric(unittest.TestCase):
    def annotate_alternates(self, chosen, alts):
        self.assertIn(chosen, alts)
        mat = ns.python_def_use_mask.match_either_name_or_global(chosen)
        if not mat:
            return chosen
        name, scope = mat.group("name"), (
            mat.group("scope") if mat.group("typ") == "&" else "0"
        )
        # print(alts)
        alts = [ns.python_def_use_mask.match_either_name_or_global(alt) for alt in alts]
        # print([x for x in alts if x])
        alts = {x.group("name") for x in alts if x}
        alts.remove(name)
        alts = sorted(alts)
        if alts:
            name = f"{name}?{'$'.join(alts)}"
        return f"const-&{name}:{scope}~Name"

    def annotate_program(
        self,
        program,
        parser=ns.python_to_python_ast,
        convert_to_python=True,
    ):
        dfa, _, fam, _ = fit_to(
            [program], parser=parser, include_type_preorder_mask=False
        )
        annotated = ns.s_exp_to_python_ast(
            ns.render_s_expression(
                ns.annotate_with_alternate_symbols(
                    ns.to_type_annotated_ns_s_exp(parser(program), dfa, "M"),
                    fam.tree_distribution_skeleton,
                    self.annotate_alternates,
                )
            )
        )
        if convert_to_python:
            return annotated.to_python()
        return annotated


class DefUseMaskTest(DefUseMaskTestGeneric):
    def test_annotate_alternate_symbols(self):
        code = self.annotate_program("x = 2; y = x; z = y")
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                x?y$z = 2
                y?x$z = x
                z?x$y = y?x
                """
            ).strip(),
        )

    def test_duplicate_lhs(self):
        code = self.annotate_program("x, x = 2, 2")
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                x, x = 2, 2
                """
            ).strip(),
        )

    def test_subscript_on_lhs(self):
        code = self.annotate_program("x = [2, 3, 4]; x[2] = x[0]; y = 2")
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                x?y = [2, 3, 4]
                x[2] = x[0]
                y?x = 2
                """
            ).strip(),
        )

    def test_attribute_on_lhs(self):
        code = self.annotate_program("x = 2; y.z = 3; x = x")
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                x?y = 2
                y?x.z = 3
                x?y = x?y
                """
            ).strip(),
        )

    def test_tuple_list_on_lhs(self):
        code = self.annotate_program("[x, y] = 2, 3; x, y = x, y; z = x")
        print(code)
        past_310 = """
        [x?y$z, y?x$z] = (2, 3)
        x?y$z, y?x$z = (x?y, y?x)
        z?x$y = x?y
        """
        up_to_310 = """
        [x?y$z, y?x$z] = (2, 3)
        (x?y$z, y?x$z) = (x?y, y?x)
        z?x$y = x?y
        """
        self.assertEqual(
            code.strip(),
            cwq(up_to_310 if sys.version_info < (3, 11) else past_310).strip(),
        )

    def test_star_tuple_on_lhs(self):
        code = self.annotate_program("x, *y = [2, 3]; x = x")
        print(code)
        past_310 = """
        x?y, *y?x = [2, 3]
        x?y = x?y
        """
        up_to_310 = """
        (x?y, *y?x) = [2, 3]
        x?y = x?y
        """
        self.assertEqual(
            code.strip(),
            cwq(up_to_310 if sys.version_info < (3, 11) else past_310).strip(),
        )

    def test_basic_import(self):
        # the 2 in front is necessary to force the import to not be pulled
        code = self.annotate_program(
            cwq(
                """
                2
                import os
                import sys as y
                from collections import defaultdict
                from collections import defaultdict as z
                x = os
                x = os
                defaultdict, os, y, z
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                2
                import os?defaultdict
                import sys as y?z
                from collections import defaultdict?os
                from collections import defaultdict as z?y
                x?defaultdict$os$y$z = os?defaultdict$y$z
                x?defaultdict$os$y$z = os?defaultdict$x$y$z
                (defaultdict?os$x$y$z, os?defaultdict$x$y$z,
                    y?defaultdict$os$x$z, z?defaultdict$os$x$y)
                """
            ).strip(),
        )

    def test_function_call(self):
        code = self.annotate_program(
            cwq(
                """
                def f(x):
                    z = x
                    return x
                y = f(2)
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                def f?x$y$z(x?f$y$z):
                    z?f$x$y = x?f
                    return x?f$z
                y?f$x$z = f(2)
                """
            ).strip(),
        )

    def test_lambda(self):
        code = self.annotate_program(
            cwq(
                """
                x = 2
                lambda y, z=x: lambda a=y: x
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                x?a$y$z = 2
                lambda y?a$x$z, z?a$x$y=x: lambda a?x$y$z=y?x$z: x?a$y$z
                """
            ).strip(),
        )

    def test_function_call_arguments(self):
        code = self.annotate_program(
            cwq(
                """
                def f(w, /, x, *y, **z):
                    return x
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                def f?w$x$y$z(w?f$x$y$z, /, x?f$w$y$z, *y?f$w$x$z, **z?f$w$x$y):
                    return x?f$w$y$z
                """
            ).strip(),
        )

    def test_single_comprehension(self):
        code = self.annotate_program(
            cwq(
                """
                a = 2
                [b for b in range(a) if b == a]
                a = a
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                a?b$range = 2
                [b?a$range for b?a$range in range?a(a?range) if b?a$range == a?b$range]
                a?b$range = a?range
                """
            ).strip(),
        )

    def test_bunch_of_comprehensions(self):
        self.maxDiff = None
        code = self.annotate_program(
            cwq(
                """
                a = 2
                [b for b in range(a)]
                (c for c in range(a))
                {c for c in range(a)}
                {d: a for d in range(a)}
                [e + f + g for e in range(a) for f in range(e) for g in range(f)]
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                a?b$c$d$e$f$g$range = 2
                [b?a$range for b?a$c$d$e$f$g$range in range?a(a?range)]
                (c?a$range for c?a$b$d$e$f$g$range in range?a(a?range))
                {c?a$range for c?a$b$d$e$f$g$range in range?a(a?range)}
                {d?a$range: a?d$range for d?a$b$c$e$f$g$range in range?a(a?range)}
                [e?a$f$g$range + f?a$e$g$range + g?a$e$f$range
                    for e?a$b$c$d$f$g$range in range?a(a?range)
                    for f?a$b$c$d$e$g$range in range?a$e(e?a$range)
                    for g?a$b$c$d$e$f$range in range?a$e$f(f?a$e$range)]
                """
            ).strip(),
        )

    def test_for(self):
        self.maxDiff = None
        code = self.annotate_program(
            cwq(
                """
                x = [2]
                for y in x:
                    y
                z = x
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                x?y$z = [2]
                for y?x$z in x:
                    y?x
                z?x$y = x?y
                """
            ).strip(),
        )

    def test_import_at_top_level(self):
        # imports at top are global so not alternated
        code = self.annotate_program("import os; import sys as y; x = os; x = os; x, y")
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                import os
                import sys as y
                x?os$y = os?y
                x?os$y = os?x$y
                (x?os$y, y?os$x)
                """
            ).strip(),
        )

    def test_class(self):
        code = self.annotate_program(
            cwq(
                """
                class A:
                    x = A
                y = A
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                class A?x$y:
                    x?A$y = A
                y?A$x = A
                """
            ).strip(),
        )

    def test_import_inside_fn(self):
        code = self.annotate_program(
            cwq(
                """
                def f():
                    from collections import defaultdict
                    return defaultdict
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                def f?defaultdict():
                    from collections import defaultdict
                    return defaultdict?f
                """
            ).strip(),
        )

    def test_function_default(self):
        code = self.annotate_program(
            cwq(
                """
                y = 2
                z = 3
                def f(x=y):
                    return x
                z = z
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                y?f$x$z = 2
                z?f$x$y = 3

                def f?x$y$z(x?f$y$z=y?f$z):
                    return x?f$y$z
                z?f$x$y = z?f$y
                """
            ).strip(),
        )

    def test_exception_named(self):
        code = self.annotate_program(
            cwq(
                """
                try:
                    x = 2
                except Exception as e:
                    x = e
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                try:
                    x?Exception$e = 2
                except Exception?x as e:
                    x?Exception$e = e?Exception$x
                """
            ).strip(),
        )

    def test_exception_unnamed(self):
        code = self.annotate_program(
            cwq(
                """
                try:
                    x = 2
                except Exception:
                    x = x
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                try:
                    x?Exception = 2
                except Exception?x:
                    x?Exception = x?Exception
                """
            ).strip(),
        )

    def test_complicated_type_annot(self):
        code = self.annotate_program(
            cwq(
                """
                x: List[Dict[str, int]] = []
                """
            )
        )
        print(code)
        self.assertEqual(
            code.strip(),
            cwq(
                """
                x: List[Dict[str, int]] = []
                """
            ).strip(),
        )

    @parameterized.expand([(i,) for i in range(20)])
    def test_realistic(self, i):
        if i in {22, 31, 41, 57, 95, 100, 106, 109, 112, 114, 119, 181, 182}:
            # forward declaration of
            # input for 22/41/100/119
            # n for 31/57/112/114
            # m for 95
            # sp for 106
            # mp for 109
            # dp for 181
            # lo for 182 [this one's weird]
            return
        example = small_set_runnable_code_examples()[i]["solution"]
        print(example)
        code = self.annotate_program(example)
        print(code)
