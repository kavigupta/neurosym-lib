from dreamcoder.program import Primitive, Program
from dreamcoder.type import arrow, baseType, tint

turtle = baseType("turtle")
tstate = baseType("tstate")
tangle = baseType("tangle")
tlength = baseType("tlength")

primitives = (
    [
        Primitive(
            "logo_UA",
            tangle,
            "",
            function_comment="Unit angle: 2 pi radians",
            alternate_names=["logo_unit_angle"],
        ),
        Primitive(
            "logo_UL",
            tlength,
            "",
            function_comment="Unit line length: 1 cm",
            alternate_names=["logo_unit_line"],
        ),
        Primitive(
            "logo_ZA",
            tangle,
            "",
            function_comment="Zero angle: 0 radians",
            alternate_names=["logo_zero_angle"],
        ),
        Primitive(
            "logo_ZL",
            tlength,
            "",
            function_comment="Zero line length: 1cm",
            alternate_names=["logo_zero_line"],
        ),
        Primitive(
            "logo_DIVA",
            arrow(tangle, tint, tangle),
            "",
            function_comment="Divide angle",
            alternate_names=["logo_divide_angle"],
        ),
        Primitive(
            "logo_MULA",
            arrow(tangle, tint, tangle),
            "",
            function_comment="Multiply angle",
            alternate_names=["logo_multiply_angle"],
        ),
        Primitive(
            "logo_DIVL",
            arrow(tlength, tint, tlength),
            "",
            function_comment="Divide line length",
            alternate_names=["logo_divide_line"],
        ),
        Primitive(
            "logo_MULL",
            arrow(tlength, tint, tlength),
            "",
            function_comment="Multiply line length",
            alternate_names=["logo_multiply_line"],
        ),
        Primitive(
            "logo_ADDA",
            arrow(tangle, tangle, tangle),
            "",
            function_comment="Add angles",
            alternate_names=["logo_add_angles"],
        ),
        Primitive(
            "logo_SUBA",
            arrow(tangle, tangle, tangle),
            "",
            function_comment="Subtract angles",
            alternate_names=["logo_subtract_angles"],
        ),
        # Primitive("logo_ADDL",  arrow(tlength,tlength,tlength), ""),
        # Primitive("logo_SUBL",  arrow(tlength,tlength,tlength), ""),
        # Primitive("logo_PU",  arrow(turtle,turtle), ""),
        # Primitive("logo_PD",  arrow(turtle,turtle), ""),
        Primitive(
            "logo_PT",
            arrow(arrow(turtle, turtle), arrow(turtle, turtle)),
            None,
            function_comment="Lift pen.",
            alternate_names=["logo_lift_pen"],
        ),
        Primitive(
            "logo_FWRT",
            arrow(tlength, tangle, turtle, turtle),
            "",
            function_comment="Move pen by length and angle.",
            alternate_names=["logo_move_pen_forward_rotate"],
        ),
        Primitive(
            "logo_GETSET",
            arrow(arrow(turtle, turtle), turtle, turtle),
            "",
            function_comment="Apply function to pen.",
            alternate_names=["logo_get_set_function_pen"],
        ),
    ]
    + [
        Primitive(
            "logo_IFTY",
            tint,
            "",
            function_comment="Integer constant of value infinity.",
        ),
        Primitive(
            "logo_epsA",
            tangle,
            "",
            function_comment="Epsilon angle",
            alternate_names=["logo_epsilon_angle"],
        ),
        Primitive(
            "logo_epsL",
            tlength,
            "",
            function_comment="Epsilon line",
            alternate_names=["logo_epsilon_line"],
        ),
        Primitive(
            "logo_forLoop",
            arrow(tint, arrow(tint, turtle, turtle), turtle, turtle),
            "ERROR: python has no way of expressing this hence you shouldn't eval on this",
            function_comment="For loop",
            alternate_names=["logo_for_loop"],
        ),
    ]
    + [
        Primitive(str(j), tint, j, function_comment=f"Integer constant of value {j}")
        for j in range(10)
    ]
)

if __name__ == "__main__":
    expr_s = "(lambda (logo_forLoop 3 (lambda (lambda (logo_GET (lambda (logo_FWRT (logo_S2L (logo_I2S 1)) (logo_S2A (logo_I2S 0)) (logo_SET $0 (logo_FWRT (logo_S2L eps) (logo_DIVA (logo_S2A (logo_I2S 2)) (logo_I2S 3)) ($1)))))))) ($0)))"
    x = Program.parse(expr_s)
    print(x)
