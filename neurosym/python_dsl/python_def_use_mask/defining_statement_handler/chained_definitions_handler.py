from ..handler import ConstructHandler, Handler
from .defining_statement_handler import DefiningStatementHandler


class ComprehensionExpressionHandler(ConstructHandler):
    """
    Handles comprehensions. This isn't a DefiningStatementHandler because
        each comprehensions' defined symbols can be used in the next
        comprehension.
    """

    # keep in mind that the ordering means generators come first

    def __init__(self, mask, defined_production_idxs, config):
        # copy the valid symbols so changes don't affect the parent
        super().__init__(mask, defined_production_idxs[:], config)
        self.defined_symbols = []

    def on_child_enter(self, position: int, symbol: int) -> Handler:
        if position == self.child_fields["generators"]:
            return GeneratorsHandler(
                self.mask, self.defined_production_idxs, self.config
            )
        return super().on_child_enter(position, symbol)

    def on_child_exit(self, position: int, symbol: int, child: Handler):
        pass

    def is_defining(self, position: int) -> bool:
        return False


class ListComprehensionHandler(ComprehensionExpressionHandler):
    name = "ListComp~E"


class SetComprehensionHandler(ListComprehensionHandler):
    name = "SetComp~E"


class GeneratorExprHandler(ListComprehensionHandler):
    name = "GeneratorExp~E"


class DictComprehensionHandler(ComprehensionExpressionHandler):
    name = "DictComp~E"


class GeneratorsHandler(Handler):
    """
    Handles a list of generators, each treated as a defining statement.
    """

    def on_child_enter(self, position: int, symbol: int) -> Handler:
        return ComprehensionHandler(
            self.mask, self.defined_production_idxs, self.config
        )

    def on_child_exit(self, position: int, symbol: int, child: Handler):
        pass

    def is_defining(self, position: int) -> bool:
        return False


class ComprehensionHandler(DefiningStatementHandler):
    name = "comprehension~C"
    targeted = ["target"]
    define_symbols_on_exit = "iter"


chained_definition_handlers = [
    ListComprehensionHandler,
    SetComprehensionHandler,
    GeneratorExprHandler,
    DictComprehensionHandler,
]
