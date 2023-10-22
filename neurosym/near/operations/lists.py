def fold_torch(func, l):  # noqa: E741
    """
    Arguments:
        func: (N, *C) -> (N, *C) -> (N, *C)
        l: List[(N, L, *C)]
    Returns:
        (N, *C)
    """
    result = l[:, 0, :]
    for i in range(1, l.shape[1]):
        result = func(result, l[:, i, :])
    return result


def map_torch(func, l):  # noqa: E741
    """
    Arguments:
        func: (N, *C1) -> (N, *C2)
        l: List[(N, L, *C1)]
    Returns:
        (N, L, *C2)
    """
    original_shape = l.shape
    reshaped = l.reshape(l.shape[0] * l.shape[1], *l.shape[2:])
    result = func(reshaped)
    return result.reshape(original_shape[0], original_shape[1], *result.shape[1:])
