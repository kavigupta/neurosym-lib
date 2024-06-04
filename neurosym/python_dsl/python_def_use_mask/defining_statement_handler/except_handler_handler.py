from ..handler import ConstructHandler, Handler


class ExceptHandlerHandler(ConstructHandler):
    """
    Handles an exception statement, which defines a name in the except block.
    """

    name = "ExceptHandler~EH"

    def on_child_enter(self, position: int, symbol: int) -> Handler:
        if self.is_defining(position):
            if self.mask.id_to_name(symbol) != "const-None~NullableName":
                self.defined_production_idxs.append(symbol)
        return super().on_child_enter(position, symbol)

    def on_child_exit(self, position: int, symbol: int, child: Handler):
        pass

    def is_defining(self, position: int) -> bool:
        return position == self.child_fields["name"]
