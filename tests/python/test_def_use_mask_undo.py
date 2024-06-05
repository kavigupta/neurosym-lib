import copy
import unittest
from textwrap import dedent

from parameterized import parameterized

import neurosym as ns

from .utils import fit_to, small_set_runnable_code_examples


class DefUseMaskTestGeneric(unittest.TestCase):
    def get_handler_except_mask(self, handler):
        return {
            k: v for k, v in handler.__dict__.items() if k not in {"mask", "config"}
        }

    def get_handlers_except_mask(self, mask):
        handlers = mask.handlers
        return [self.get_handler_except_mask(handler) for handler in handlers]

    def annotate_program(self, program, with_undo, with_undo_exit):
        def replace_node_midstream(s_exp, mask, position, alts):
            if with_undo:
                for alt in alts:
                    undo = mask.on_entry(position, alt)
                    undo()
            if with_undo_exit:
                for alt in alts:
                    undo_entry = mask.on_entry(position, alt)
                    print("*" * 80)
                    print("handlers", mask.handlers)
                    last_handler = copy.deepcopy(self.get_handlers_except_mask(mask))
                    print("copy", last_handler)
                    undo_exit = mask.on_exit(position, alt)
                    print("after exit", self.get_handlers_except_mask(mask))
                    undo_exit()
                    print(
                        "after undo exit",
                        self.get_handlers_except_mask(mask),
                    )
                    self.assertEqual(
                        last_handler,
                        self.get_handlers_except_mask(mask),
                    )
                    undo_entry()

            return s_exp

        dfa, _, fam, _ = fit_to(
            [program], parser=ns.python_to_python_ast, include_type_preorder_mask=False
        )
        td = fam.tree_distribution_skeleton
        result = list(
            ns.collect_preorder_symbols(
                ns.to_type_annotated_ns_s_exp(
                    ns.python_to_python_ast(program), dfa, "M"
                ),
                fam.tree_distribution_skeleton,
                replace_node_midstream=replace_node_midstream,
            )
        )
        result = [
            (ns.render_s_expression(s_exp), [td.symbols[i][0] for i in alts])
            for s_exp, alts, _ in result
        ]
        # print(result)
        return result

    def assertUndoHasNoEffect(self, program):
        self.maxDiff = None
        with_undo = self.annotate_program(program, with_undo=True, with_undo_exit=False)
        with_undo_exit = self.annotate_program(
            program, with_undo=False, with_undo_exit=True
        )
        without_undo = self.annotate_program(
            program, with_undo=False, with_undo_exit=False
        )
        self.assertEqual(with_undo, without_undo)
        self.assertEqual(with_undo_exit, without_undo)


class DefUseMaskTest(DefUseMaskTestGeneric):
    def test_annotate_alternate_symbols(self):
        self.assertUndoHasNoEffect("x = 2; y = x; z = y")

    def test_for(self):
        self.assertUndoHasNoEffect(
            dedent(
                r"""
                for i in range(2):
                    x = 2
                print(i)
                """
            )
        )

    def test_for_then_subscript(self):
        self.assertUndoHasNoEffect(
            dedent(
                r"""
                for i in range(2):
                    x = 2
                x[0] = 2
                """
            )
        )

    def test_temp(self):
        self.assertUndoHasNoEffect(
            dedent(
                r"""
                for u in 2:
                    pass
                x[0] = 2
                """
            )
        )

    @parameterized.expand([(i,) for i in range(50)])
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
        self.assertUndoHasNoEffect(example)
