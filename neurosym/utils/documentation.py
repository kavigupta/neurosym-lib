def internal_only(func):
    """Decorator to mark a function as deliberately not documented as it is not intended for public use."""
    func.__internal_only__ = True
    return func


internal_only(internal_only)
