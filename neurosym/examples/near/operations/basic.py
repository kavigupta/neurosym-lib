import torch


def ite_torch(condition, if_true, if_else):
    def _ite(*args):
        cond = torch.sigmoid(condition(*args))
        true_val = if_true(*args)
        false_val = if_else(*args)
        if len(cond.shape) == 2 and len(true_val.shape) == 3:
            cond = cond.unsqueeze(1)
        cond = cond.expand_as(true_val)
        return cond * true_val + (1 - cond) * false_val

    return _ite
