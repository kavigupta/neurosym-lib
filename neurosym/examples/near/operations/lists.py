def fold_torch(func, lst):  # noqa: E741
    """
    Runs a fold operation on a list of tensors, handling the batch dimension.

    :param func: ``((N, *C), (N, *C)) -> (N, *C)``
    :param lst: ``(N, L, *C)``
    :returns: ``(N, *C)``
    """
    result = lst[:, 0, :]
    for i in range(1, lst.shape[1]):
        result = func(result, lst[:, i, :])
    return result


def map_torch(func, lst):  # noqa: E741
    """

    Runs a map operation on a list of tensors, handling the batch dimension.

    :param func: ``(N, *C1) -> (N, *C2)``
    :param lst: ``(N, L, *C1)``
    :returns: ``(N, L, *C2)``
    """
    original_shape = lst.shape
    reshaped = lst.reshape(lst.shape[0] * lst.shape[1], *lst.shape[2:])
    result = func(reshaped)
    return result.reshape(original_shape[0], original_shape[1], *result.shape[1:])
