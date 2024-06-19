from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from neurosym.python_dsl.python_ast_tools import fields_for_node

from .names import match_either_name_or_global
from .special_case_symbol_predicate import SpecialCaseSymbolPredicate


class Handler(ABC):
    """
    Corresponds to a given node in the ``PythonAST``. Keeps track of a list of defined
    production indices, in order of definition, along with a reference to the mask, and
    the configuration.

    The handler is responsible for determining whether a given context is defining or not,
    and for computing the mask for a given set of names.

    The handler is also responsible for determining the handler for a given child, and for
    performing tasks related to the child when it is exited.

    :param mask: The ``DefUsePreorderMask`` that is being used, this is a circular reference
        used for convenience.
    :param defined_production_idxs: The list of defined production indices, in order of
        definition.
    :param config: The configuration for the mask.
    """

    def __init__(self, mask, defined_production_idxs, config):
        assert isinstance(defined_production_idxs, list)
        self.mask = mask
        self.defined_production_idxs = defined_production_idxs
        self.config = config

    def on_child_enter(
        self, position: int, symbol: int
    ) -> Tuple["Handler", Callable[[], None]]:
        """
        When a child is entered, this method is called to determine the handler for the child.
        This method can also perform tasks related to entering the child. By default, this
        method returns the default handler for the child, and does nothing else.

        This method can be undone by calling the returned function.

        :param position: The position in the s-expression.
        :param symbol: The symbol of the child (index into a grammar's symbols list).
            Note: this can include variables, e.g., const-&x:0-Name,
            but it can also include production symbols like Assign~S
            or other leaves like const-i2~Const.

        :return: ``(child_handler, undo)``: The handler for the child and a function
            that can be called to undo the changes made by this function.
        """
        return (
            default_handler(
                position,
                symbol,
                self.mask,
                self.currently_defined_indices(),
                self.config,
            ),
            lambda: None,
        )

    def on_child_exit(
        self, position: int, symbol: int, child: "Handler"
    ) -> Callable[[], None]:
        """
        When a child is exited, this method is called to perform tasks related to
        processing the information from the child. By default, this method does nothing.

        :param position: The position in the s-expression.
        :param symbol: The symbol of the child. See ``on_child_enter`` for more information.
        :param child: The handler for the child.

        :return: A function that can be called to undo the changes made by this function.
        """
        del position, symbol, child
        return lambda: None

    def currently_defined_indices(self) -> list[int]:
        """
        Returns the list of currently defined symbols, in order of definition. This is
        important because we need to know which symbols are currently defined for some
        purposes, e.g., when using imperative de bruijn indices.
        """
        assert isinstance(self.defined_production_idxs, list)
        return self.defined_production_idxs

    @abstractmethod
    def is_defining(self, position: int) -> bool:
        """
        Returns whether the context at the given position is defining.

        E.g., for an Assign node, the left-hand side is defining, and
        the right-hand side is not. This is important because for
        defining contexts, we do not need to use a previously
        defined variable, so new variables will not be masked out.

        :param position: The position in the s-expression.
        """

    def currently_defined_names(self):
        """
        Return the set of currently defined names. Note that this
        isn't the set of production symbols like ``const-&x:0-Name``,
        but rather a set of names like ``x``.
        """
        names = set()
        for symbol in self.currently_defined_indices():
            sym = self.mask.id_to_name(symbol)
            mat = match_either_name_or_global(sym)
            if not mat:
                continue
            names.add(mat.group("name"))
        return names

    def target_child(
        self, position: int, symbol: int
    ) -> Tuple["Handler", Callable[[], None]]:
        """
        Return a handler for a target. This is used when we are processing a target,
        e.g., the left-hand side of an assignment. By default, this method returns the
        default handler for the child, and does nothing else.

        :param position: The position in the s-expression.
        :param symbol: The symbol of the child. See ``on_child_enter`` for more information.
        """
        # pylint: disable=cyclic-import
        from .target_handler import create_target_handler

        return (
            create_target_handler(
                position,
                symbol,
                self.mask,
                self.currently_defined_indices(),
                self.config,
            ),
            lambda: None,
        )

    def _matches(self, names, symbol_id, idx_to_name, special_case_predicates):
        """
        Whether or not the symbol matches the names.
        """

        for pred in special_case_predicates:
            if pred.applies(symbol_id):
                return pred.compute(symbol_id, names)
        if idx_to_name[symbol_id] is None:
            return True
        return idx_to_name[symbol_id] in names

    def compute_mask(
        self,
        position: int,
        symbols: List[int],
        idx_to_name: List[str],
        special_case_predicates: List[SpecialCaseSymbolPredicate],
    ):
        """
        Compute the mask for the given names. If the context is defining,
        then all symbols are valid. Otherwise, only the symbols that
        match the names are valid.

        :param position: The position in the s-expression.
        :param symbols: The list of symbols to compute the mask for.
        :param idx_to_name: The list of names for the symbols.
        :param special_case_predicates: The list of special case predicates to use.
            A special case predicate is a way to extend the behavior of the handler
            by providing a custom predicate for a given symbol as to whether
            it should be considered valid or not.

        :return: A list of booleans, where ``True`` indicates that the symbol is valid.
        """
        assert isinstance(special_case_predicates, list) and all(
            isinstance(x, SpecialCaseSymbolPredicate) for x in special_case_predicates
        )
        if self.is_defining(position):
            return [True] * len(symbols)
        names = set(self.currently_defined_names())
        return [
            self._matches(names, symbol, idx_to_name, special_case_predicates)
            for symbol in symbols
        ]


class ConstructHandler(Handler):
    """
    Base class for Handlers that correspond to a specific construct in the AST.

    :field name: The name of the construct. This must be the name of a subclass of
        ``ast.AST``. This is used to determine the child fields of the construct.
    """

    # must be overridden in subclasses, represents the name of the construct
    name: str = None

    def __init__(self, mask, defined_production_idxs, config):
        super().__init__(mask, defined_production_idxs, config)
        assert isinstance(self.name, str)
        self.child_fields = {
            field: i for i, field in enumerate(fields_for_node(self.name))
        }


class DefaultHandler(Handler):
    """
    The default handler for a given symbol. This handler is used
    when a construct does nothing interesting with symbols. Effectively, just a
    passthrough handler.
    """

    def is_defining(self, position: int) -> bool:
        return False


def default_handler(
    position: int, symbol: int, mask, defined_production_idxs, config
) -> Handler:
    """
    Compute the default handler for a given symbol. This chooses the type
    of the handler based on the head symbol.

    :param position: The position in the s-expression.
    :param symbol: The symbol of the construct.
    :param mask: The mask being used.
    :param defined_production_idxs: The list of currently defined production indices.
    :param config: The configuration for the mask.
    """
    # pylint: disable=cyclic-import
    from .defining_statement_handler import defining_statement_handlers

    assert isinstance(defined_production_idxs, list)

    symbol = mask.id_to_name(symbol)
    pulled = config.pull_handler(position, symbol, mask, defined_production_idxs)
    if pulled is not None:
        return pulled

    return defining_statement_handlers().get(symbol, DefaultHandler)(
        mask, defined_production_idxs, config
    )


class HandlerPuller(ABC):
    """
    Represents a class that can pull a handler for a given symbol. This is
    used to extend the behavior of the DefUsePreorderMask by providing
    a space where hooks can be added to change the behavior of the mask
    on newly defined symbols.
    """

    @abstractmethod
    def pull_handler(
        self,
        position: int,
        symbol: int,
        mask,
        defined_production_idxs,
        config,
        handler_fn: Callable,
    ) -> Handler:
        """
        Pull a handler for the given symbol, position, etc.

        :param position: The position in the s-expression.
        :param symbol: The symbol of the construct.
        :param mask: The mask being used.
        :param defined_production_idxs: The list of currently defined production indices.
        :param config: The configuration for the mask.
        :param handler_fn: The function to call to get the default handler at the given location.
            This is provided because the default handler is not always the ``default_handler``
            function's result, i.e., when in a target context.

        :return: The handler to use for the given symbol.
        """
