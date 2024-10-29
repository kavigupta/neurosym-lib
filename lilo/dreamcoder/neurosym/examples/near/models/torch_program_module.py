from torch import nn


class TorchProgramModule(nn.Module):
    """
    Module that wraps a program into a torch.nn.Module. The program is initialized,
    and the contained modules are added to the module list.

    :param dsl: The DSL that the program is written in.
    :param program: The program to wrap.
    """

    def __init__(self, dsl, program):
        super().__init__()
        self.dsl = dsl
        self.program = program
        self.initalized_program = dsl.initialize(program)
        self.contained_modules = nn.ModuleList(
            list(self.initalized_program.all_state_values())
        )

    def forward(self, *args, environment):
        return self.dsl.compute(self.initalized_program, environment)(*args)
