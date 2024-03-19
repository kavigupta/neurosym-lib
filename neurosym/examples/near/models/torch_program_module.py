from torch import nn
from neurosym.programs.s_expression_render import render_s_expression


class TorchProgramModule(nn.Module):
    def __init__(self, dsl, program):
        super().__init__()
        self.dsl = dsl
        self.program = program
        self.initialized_program = dsl.initialize(program)
        self.contained_modules = nn.ModuleList(
            list(self.initialized_program.all_state_values())
        )

    def forward(self, *args):
        return self.dsl.compute(self.initialized_program)(*args)

    def __repr__(self):
        super_rep = super().__repr__()
        return f"{super_rep}\n w/ SExpr: {render_s_expression(self.program)}"
