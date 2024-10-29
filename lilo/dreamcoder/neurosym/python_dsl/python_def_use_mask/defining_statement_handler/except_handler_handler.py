from typing import Callable, Tuple

from neurosym.program_dist.tree_distribution.preorder_mask.undos import chain_undos
from neurosym.utils.documentation import internal_only

from ..handler import ConstructHandler, Handler


@internal_only
class ExceptHandlerHandler(ConstructHandler):
    """
    Handles an exception statement, which defines a name in the except block.
    """

    name = "ExceptHandler~EH"

    def on_child_enter(
        self, position: int, symbol: int
    ) -> Tuple[Handler, Callable[[], None]]:
        undos = []
        if self.is_defining(position):
            if self.mask.id_to_name(symbol) != "const-None~NullableName":
                self.defined_production_idxs.append(symbol)
                undos.append(self.defined_production_idxs.pop)
        handler, undo = super().on_child_enter(position, symbol)
        undos.append(undo)
        return handler, chain_undos(undos)

    def is_defining(self, position: int) -> bool:
        return position == self.child_fields["name"]
