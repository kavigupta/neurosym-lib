from neurosym.utils.documentation import internal_only


@internal_only
def defining_statement_handlers():
    """
    Map from statement name to handler class for defining statements.
    """
    # pylint: disable=cyclic-import
    from .chained_definitions_handler import chained_definition_handlers
    from .defining_construct_handler import defining_construct_handler
    from .defining_statement_handler import (
        AnnAssignHandler,
        AssignHandler,
        ForHandler,
        ImportFromHandler,
        ImportHandler,
        LambdaHandler,
    )
    from .except_handler_handler import ExceptHandlerHandler

    return {
        x.name: x
        for x in [
            AssignHandler,
            AnnAssignHandler,
            ImportHandler,
            ImportFromHandler,
            ForHandler,
            LambdaHandler,
            ExceptHandlerHandler,
            *defining_construct_handler,
            *chained_definition_handlers,
        ]
    }
