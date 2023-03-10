import cupy as cp


def multi_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -cp.sum(cp.log(y[cp.arange(batch_size), t] + 1e-7))/batch_size


def binary_cross_entropy(predict, true):
    return -(true*cp.log(predict + 1e-7)+(1-true)*cp.log(1-predict+1e-7))

