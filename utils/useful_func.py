from time import time


def print_result(current, total, begin):
    lis = ['[' if i == 0 else ']' if i == 21 else ' ' for i in range(22)]
    index = int((current+1)/total*20)
    percentage = format((current+1)*100 / total, '.2f')
    if 0 <= index < 20:
        pass
    else:
        index = 20
    if index > 0:
        for i in range(1,index+1):
            lis[i] = u'\u25A0'
        string = ''.join(lis)
        time1 = time() - begin
        print(f'\r{string} {percentage}% Time: {time1:.3f}s', end='', flush=True)
    else:
        string = ''.join(lis)
        time1 = time() - begin
        print(f'\r{string} {percentage}% Time: {time1:.3f}s', end='', flush=True)


def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)
        for i in range(0, L - 1):
            for j in range(i + 1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                elif params[i].ndim == 2 and params[j].ndim == 2 and params[
                        i].T.shape == params[j].shape and np.all(
                            params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                if find_flg: break
            if find_flg: break
        if not find_flg: break

    return params, grads