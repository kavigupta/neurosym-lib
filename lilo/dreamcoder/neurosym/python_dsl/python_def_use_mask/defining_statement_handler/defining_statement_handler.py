from typing import Callable, Tuple

from neurosym.program_dist.tree_distribution.preorder_mask.undos import (
    chain_undos,
    remove_last_n_elements,
)
from neurosym.utils.documentation import internal_only

from ..handler import ConstructHandler, Handler


class DefiningStatementHandler(ConstructHandler):
    """
    Represents a statement that defines symbols. This handler contains a field
    ``defined_symbols`` that contains the symbols defined by the statement, and
    then defines the symbols in the parent frame when the given child is exited.

    :field targeted: The fields in the construct that define the symbols. Should be
        defined in the subclass.
    :field define_symbols_on_exit: The field in the construct after which the
        symbols are defined.
        Should be defined in the subclass.
    """

    # these fields must be defined in the subclass
    targeted: list[str] = None
    # the field after which the symbols are defined
    define_symbols_on_exit: str = None

    def __init__(self, mask, defined_production_idxs, config):
        super().__init__(mask, defined_production_idxs, config)
        assert isinstance(self.name, str)
        assert isinstance(self.targeted, list)
        assert isinstance(self.define_symbols_on_exit, str)
        self._targeted_positions = [self.child_fields[child] for child in self.targeted]
        self.defined_symbols = []

    def on_child_enter(
        self, position: int, symbol: int
    ) -> Tuple[Handler, Callable[[], None]]:
        if position in self._targeted_positions:
            return self.target_child(position, symbol)
        return super().on_child_enter(position, symbol)

    def on_child_exit(
        self, position: int, symbol: int, child: Handler
    ) -> Callable[[], None]:
        undos = []
        if position in self._targeted_positions:
            if len(set(child.defined_symbols)) != len(child.defined_symbols):
                raise ValueError(
                    f"Duplicate symbols defined in child {child}: {child.defined_symbols}"
                )
            self.defined_symbols += child.defined_symbols
            undos.append(
                remove_last_n_elements(self.defined_symbols, len(child.defined_symbols))
            )
        if position == self.child_fields[self.define_symbols_on_exit]:
            current = set(self.defined_production_idxs)
            additional_symbols = [x for x in self.defined_symbols if x not in current]
            self.defined_production_idxs += additional_symbols
            undos.append(
                remove_last_n_elements(
                    self.defined_production_idxs, len(additional_symbols)
                )
            )
        undos.append(super().on_child_exit(position, symbol, child))
        return chain_undos(undos)

    def is_defining(self, position: int) -> bool:
        return position in self._targeted_positions


@internal_only
class ChildFrameCreatorHandler(DefiningStatementHandler):
    def __init__(self, mask, defined_production_idxs, config):
        self.original_defined_production_idxs = defined_production_idxs
        super().__init__(mask, list(defined_production_idxs), config)


@internal_only
class AssignHandler(DefiningStatementHandler):
    name = "Assign~S"
    targeted = ["targets"]
    define_symbols_on_exit = "type_comment"


@internal_only
class AnnAssignHandler(DefiningStatementHandler):
    name = "AnnAssign~S"
    targeted = ["target"]
    define_symbols_on_exit = "simple"


@internal_only
class ForHandler(DefiningStatementHandler):
    name = "For~S"
    targeted = ["target"]
    define_symbols_on_exit = "iter"


@internal_only
class ImportHandler(DefiningStatementHandler):
    name = "Import~S"
    targeted = ["names"]
    define_symbols_on_exit = "names"


@internal_only
class ImportFromHandler(ImportHandler):
    name = "ImportFrom~S"


@internal_only
class LambdaHandler(ChildFrameCreatorHandler):
    name = "Lambda~E"
    targeted = ["args"]
    define_symbols_on_exit = "args"
