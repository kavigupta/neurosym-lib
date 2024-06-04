from ..handler import Handler
from .defining_statement_handler import ChildFrameCreatorHandler


class DefiningConstructHandler(ChildFrameCreatorHandler):
    """
    Handles defining constructs like function and class definitions.

    These are constructs that have a child frame defined as the body of the construct,
        as well as a name that is defined in the parent frame.
    """

    # these fields must be defined in the subclass
    construct_name_field: str = None

    def __init__(self, mask, defined_production_idxs, config):
        super().__init__(mask, defined_production_idxs, config)
        assert isinstance(self.construct_name_field, str)

    def on_child_enter(self, position: int, symbol: int) -> Handler:
        if self.is_construct_name_field(position):
            return self.target_child(position, symbol)
        return super().on_child_enter(position, symbol)

    def on_child_exit(self, position: int, symbol: int, child: Handler):
        if self.is_construct_name_field(position):
            print(self, [self.mask.id_to_name(x) for x in child.defined_symbols])
            for idx_list in (
                self.original_defined_production_idxs,
                self.defined_production_idxs,
            ):
                idx_list += [x for x in child.defined_symbols if x not in idx_list]
        super().on_child_exit(position, symbol, child)

    def is_construct_name_field(self, position):
        return (
            self.construct_name_field is not None
            and position == self.child_fields[self.construct_name_field]
        )

    def is_defining(self, position: int) -> bool:
        if position == self.child_fields[self.construct_name_field]:
            return True
        return super().is_defining(position)


class FuncDefHandler(DefiningConstructHandler):
    name = "FunctionDef~S"
    targeted = ["args"]
    define_symbols_on_exit = "args"
    construct_name_field = "name"


class ClassDefHandler(DefiningConstructHandler):
    name = "ClassDef~S"
    targeted = []
    define_symbols_on_exit = "decorator_list"
    construct_name_field = "name"


defining_construct_handler = [FuncDefHandler, ClassDefHandler]
