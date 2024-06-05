def chain_undos(undos):
    """
    Run a series of undo functions. The undos are run in reverse order,
        so the last undo function is run first.
    """

    def undo():
        for undo in undos[::-1]:
            undo()

    return undo


def remove_last_n_elements(lst, n):
    """
    Remove the last n elements from a list.
    """

    def run():
        if n > 0:
            del lst[-n:]

    return run
