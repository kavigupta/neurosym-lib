import torch


def ite_torch(condition, if_true, if_else):
    """
    Computes a differentiable if-then-else operation.

    :param condition: A function that computes the condition. This is
        expected to return a tensor of shape ``(N, *C1)``, and the output
        will be broadcast* to the shape of the other arguments. The
        condition will have a sigmoid applied to it to put it in the
        range [0, 1], so positive values will be interpreted as true,
        and negative values as false.

    :param if_true: A function that computes the value if the condition
        is true. This is expected to return a tensor of shape ``(N, *C2)``.

    :param if_else: A function that computes the value if the condition
        is false. This is expected to return a tensor of shape ``(N, *C2)``.

    \\*Broadcasting here means that the condition tensor must have a shape
    that is a prefix of the shape of the other tensors, with potential 1s
    at the end. For example, if the condition has shape (N, 1), and the
    other tensors have shape (N, M), then the condition will be broadcast
    to shape (N, M) by repeating the condition M times.
    """

    def _ite(*args):
        cond = torch.sigmoid(condition(*args))
        true_val = if_true(*args)
        false_val = if_else(*args)
        cond = _normalize_shapes(cond, true_val, false_val)
        if len(cond.shape) == len(true_val.shape) - 1:
            cond = cond.unsqueeze(-1)
        cond = cond.expand_as(true_val)
        return cond * true_val + (1 - cond) * false_val

    return _ite


def _normalize_shapes(cond, *tensors):
    """
    Broadcasts the condition tensor to the shape of the other tensors.

    cond must have a shape that is a prefix of the shape of all the other tensors, with potential
    1s at the end
    """
    tensor_shapes = [t.shape for t in tensors]
    if not all(ts == tensor_shapes[0] for ts in tensor_shapes):
        raise ValueError(
            f"All value tensors must have the same shape, got {tensor_shapes}"
        )

    target_shape = tensor_shapes[0]
    cond_shape = list(cond.shape)
    while cond_shape and cond_shape[-1] == 1:
        cond_shape.pop()
    if not target_shape[: len(cond_shape)] == tuple(cond_shape):
        raise ValueError(
            f"Condition shape {cond.shape} is not a prefix of value shape {target_shape}"
        )
    cond_shape = list(cond_shape) + [1] * (len(target_shape) - len(cond_shape))
    return cond.view(*cond_shape)
