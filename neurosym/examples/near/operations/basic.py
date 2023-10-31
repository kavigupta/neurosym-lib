import torch


def ite_torch(condition, if_true, if_else):
    def _ite(*args):
        cond = torch.sigmoid(condition(*args))
        return cond * if_true(*args) + (1 - cond) * if_else(*args)

    return _ite
