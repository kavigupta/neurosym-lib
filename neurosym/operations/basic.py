

def ite(condition, if_true, if_else):
    def _ite(*args):
        if condition(*args):
            return if_true(*args)
        else:
            return if_else(*args)

    return _ite