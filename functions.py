import cupy as np


def print_result(current, total, begin, end):
    lis = ['[' if i == 0 else ']' if i == 21 else ' ' for i in range(22)]
    index = int(current / total * 20)
    percentage = format(current * 100 / total, '.2f')
    if 0 <= index < 20:
        pass
    else:
        index = 20
    if index > 0:
        for i in range(1, index + 1):
            lis[i] = u'\u25A0'
        string = ''.join(lis)
        time = end - begin
        print(f'\r{string} {percentage}% Time: {time:.3f}s',
              end='',
              flush=True)
    else:
        string = ''.join(lis)
        time = end - begin
        print(f'\r{string} {percentage}% Time: {time:.3f}s',
              end='',
              flush=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size