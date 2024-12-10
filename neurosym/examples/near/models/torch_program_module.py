from torch import nn


class TorchProgramModule(nn.Module):
    """
    Module that wraps a program into a torch.nn.Module. The program is initialized,
    and the contained modules are added to the module list.

    :param dsl: The DSL that the program is written in.
    :param initialized_program: The initialized program to wrap.
    """

    def __init__(self, dsl, program, initialized_program=None):
        super().__init__()
        self.dsl = dsl
        self.initalized_program = (
            dsl.initialize(program)
            if initialized_program is None
            else initialized_program
        )
        self.contained_modules = nn.ModuleList(
            list(self.initalized_program.all_state_values())
        )

    def forward(self, *args, environment):
        return self.dsl.compute(self.initalized_program, environment)(*args)
