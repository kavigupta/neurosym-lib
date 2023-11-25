from torch import nn


class TorchProgramModule(nn.Module):
    def __init__(self, dsl, program):
        super().__init__()
        self.dsl = dsl
        self.initalized_program = dsl.initialize(program)
        self.contained_modules = nn.ModuleList(
            list(self.initalized_program.all_state_values())
        )

    def forward(self, *args):
        return self.dsl.compute(self.initalized_program)(*args)
