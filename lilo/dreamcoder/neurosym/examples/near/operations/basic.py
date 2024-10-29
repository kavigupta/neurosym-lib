import torch


def ite_torch(condition, if_true, if_else):
    """
    Computes a differentiable if-then-else operation.

    :param condition: A function that computes the condition. This is
        expected to return a tensor of shape ``(N, *C1)``, and the output
        will be broadcast to the shape of the other arguments. The
        condition will have a sigmoid applied to it to put it in the
        range [0, 1], so positive values will be interpreted as true,
        and negative values as false.

    :param if_true: A function that computes the value if the condition
        is true. This is expected to return a tensor of shape ``(N, *C2)``.

    :param if_else: A function that computes the value if the condition
        is false. This is expected to return a tensor of shape ``(N, *C2)``.
    """

    def _ite(*args):
        cond = torch.sigmoid(condition(*args))
        true_val = if_true(*args)
        false_val = if_else(*args)
        if len(cond.shape) == len(true_val.shape) - 1:
            cond = cond.unsqueeze(1)
        cond = cond.expand_as(true_val)
        return cond * true_val + (1 - cond) * false_val

    return _ite
