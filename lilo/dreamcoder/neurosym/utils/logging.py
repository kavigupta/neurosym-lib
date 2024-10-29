from neurosym.utils.documentation import internal_only


@internal_only
def log(*args, **kwargs):
    """
    Like print, but does not cause the test to complain.
    """
    print(*args, **kwargs)
