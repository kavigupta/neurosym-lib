from typing import Callable, Tuple

from neurosym.program_dist.tree_distribution.preorder_mask.undos import (
    remove_last_n_elements,
)
from neurosym.python_dsl.names import PYTHON_DSL_SEPARATOR
from neurosym.utils.documentation import internal_only

from .handler import ConstructHandler, Handler


def create_target_handler(
    position: int, root_symbol: int, mask, defined_production_idxs, config
):
    """
    Create a target handler for the given root symbol. A target handler collects
    the assigned targets in the given expression.

    :param position: The position of the child in the parent.
    :param root_symbol: The root symbol of the expression.
    :param mask: The mask.
    :param defined_production_idxs: The defined production indices.
    :param config: The configuration.
    """
    targets_map = {
        "Name~L": NameTargetHandler,
        "arg~A": ArgTargetHandler,
        "alias~alias": AliasTargetHandler,
        "const-None~A": NonCollectingTargetHandler,
        "Subscript~L": NonCollectingTargetHandler,
        "Attribute~L": NonCollectingTargetHandler,
        "Tuple~L": TupleLHSHandler,
        "List~L": ListLHSHandler,
        "_starred_content~L": PassthroughLHSHandler,
        "_starred_starred~L": PassthroughLHSHandler,
        "Starred~L": StarredHandler,
        "arguments~As": ArgumentsHandler,
    }

    symbol = root_symbol
    symbol = mask.id_to_name(symbol)

    if symbol.startswith("const") and symbol.split(PYTHON_DSL_SEPARATOR)[-1] in {
        "Name",
        "NullableName",
        "NameStr",
        "NullableNameStr",
    }:
        return SymbolTargetHandler(mask, defined_production_idxs, config, root_symbol)

    pulled = config.pull_handler(
        position,
        symbol,
        mask,
        defined_production_idxs,
        handler_fn=create_target_handler,
    )
    if pulled is not None:
        return pulled

    if symbol.startswith("list"):
        return PassthroughLHSHandler(mask, defined_production_idxs, config)
    return targets_map[symbol](mask, defined_production_idxs, config)


class TargetHandler(Handler):
    """
    Handler that collects targets, stored in defined_symbols. This is used for
    LHS values that define symbols. It collects targets from any child frame
    by default.

    :field defined_symbols: The symbols defined by the construct.
    """

    def __init__(self, mask, defined_production_idxs, config):
        super().__init__(mask, defined_production_idxs, config)
        self.defined_symbols = []

    def on_child_exit(
        self, position: int, symbol: int, child: Handler
    ) -> Callable[[], None]:
        if hasattr(child, "defined_symbols"):
            symbols = [
                x for x in child.defined_symbols if x not in self.defined_symbols
            ]
            self.defined_symbols += symbols
            return remove_last_n_elements(self.defined_symbols, len(symbols))
        return lambda: None


@internal_only
class TargetConstructHandler(TargetHandler, ConstructHandler):
    def __init__(self, mask, defined_production_idxs, config):
        TargetHandler.__init__(self, mask, defined_production_idxs, config)
        ConstructHandler.__init__(self, mask, defined_production_idxs, config)


@internal_only
class PassthroughLHSHandler(TargetHandler):
    """
    Pass through handler that does not collect any information,
        instead it just targets the children at the given indices.

    If indices is None, it will target all children.
    """

    def on_child_enter(
        self, position: int, symbol: int
    ) -> Tuple[Handler, Callable[[], None]]:
        if self.is_defining(position):
            return self.target_child(position, symbol)
        return super().on_child_enter(position, symbol)

    def is_defining(self, position: int) -> bool:
        return True


@internal_only
class PassthroughLHSConstructHandler(PassthroughLHSHandler, TargetConstructHandler):
    """
    Pass through handler that is also a construct handler.
    """

    use_fields: list[str] = None

    def __init__(self, mask, defined_production_idxs, config):
        PassthroughLHSHandler.__init__(self, mask, defined_production_idxs, config)
        TargetConstructHandler.__init__(self, mask, defined_production_idxs, config)

        self.indices = [self.child_fields[x] for x in self.use_fields]

    def is_defining(self, position: int) -> bool:
        return self.indices is None or position in self.indices


@internal_only
class NonCollectingTargetHandler(PassthroughLHSHandler):
    """
    This is for LHS values where nothing is actually being defined
        (e.g., Subscript, Attribute, etc.)
    """

    def is_defining(self, position: int) -> bool:
        return False


@internal_only
class SymbolTargetHandler(TargetHandler):
    """
    Target handler for symbols. Since symbols have no children, this takes in the symbol
        and sets it up in the defined symbols in __init__.

    Do not keep the symbol if it is a const-None~NullableNameStr.
    """

    def __init__(self, mask, defined_production_idxs, config, symbol):
        super().__init__(mask, defined_production_idxs, config)
        if self.mask.id_to_name(symbol) != "const-None~NullableNameStr":
            self.defined_symbols = [symbol]

    def on_child_enter(
        self, position: int, symbol: int
    ) -> Tuple[Handler, Callable[[], None]]:
        raise NotImplementedError("symbols should not have children.")

    def on_child_exit(
        self, position: int, symbol: int, child: Handler
    ) -> Callable[[], None]:
        raise NotImplementedError("symbols should not have children.")

    def is_defining(self, position: int) -> bool:
        return True


@internal_only
class StarredHandler(PassthroughLHSConstructHandler):
    """
    Handles starred expressions.
    """

    name = "Starred~L"
    use_fields = ["value"]


@internal_only
class ArgumentsHandler(PassthroughLHSConstructHandler):
    """
    Handles arguments.
    """

    name = "arguments~As"
    use_fields = ("posonlyargs", "args", "vararg", "kwonlyargs", "kwarg")


@internal_only
class TupleLHSHandler(PassthroughLHSConstructHandler):
    """
    This is for LHS values where nothing is actually being defined
        (e.g., Subscript, Attribute, etc.)
    """

    name = "Tuple~L"
    use_fields = ["elts"]


@internal_only
class ListLHSHandler(TupleLHSHandler):
    """
    Handler for list LHS values.
    """

    name = "List~L"


@internal_only
class NameTargetHandler(TargetConstructHandler):
    """
    Handler for Name values, as well as other constructs containing symbols.
        Always keeps only the last defined symbol.
    """

    name = "Name~Name"
    # will select the last of these that is defined
    name_nodes = {"id"}

    def on_child_enter(
        self, position: int, symbol: int
    ) -> Tuple[Handler, Callable[[], None]]:
        if self.is_defining(position):
            return self.target_child(position, symbol)
        return super().on_child_enter(position, symbol)

    def on_child_exit(
        self, position: int, symbol: int, child: Handler
    ) -> Callable[[], None]:
        if hasattr(child, "defined_symbols") and child.defined_symbols:
            previous = self.defined_symbols
            self.defined_symbols = child.defined_symbols
            return lambda: setattr(self, "defined_symbols", previous)
        return lambda: None

    def is_defining(self, position: int) -> bool:
        return any(position == self.child_fields[x] for x in self.name_nodes)


@internal_only
class ArgTargetHandler(NameTargetHandler):
    name = "arg~arg"
    name_nodes = {"arg"}


@internal_only
class AliasTargetHandler(NameTargetHandler):
    name = "alias~alias"
    name_nodes = {"name", "asname"}
