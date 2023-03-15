import cupy as cp


class Add:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self):
        grad1, grad2 = get_grads(self.x, self.y)
        return grad1 + grad2


class Minus:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self):
        grad1, grad2 = get_grads(self.x, self.y)
        return grad1 - grad2


class Multi:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self):
        grad1, grad2 = get_grads(self.x, self.y)

        result1, result2 = get_values(self.x, self.y)

        return grad1*result2 + result1*grad2, result1*result2


class Divide:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self):
        grad1, grad2 = get_grads(self.x, self.y)
        result1, result2 = get_values(self.x, self.y)
        return (grad1*result2 - result1*grad2)/cp.square(result2), result1/result2


class expe:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.exp(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class exp:
    def __init__(self, x, power):
        self.x = power
        self.val = get_value(x)

    def result(self):
        return cp.power(self.val, self.x)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class sin:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        if isinstance(val, Multi):
            return val.get_grad()[1]
        return cp.sin(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class cos:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        if isinstance(val, Multi):
            return cp.cos(val.get_grad()[1])
        return cp.cos(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other).get_grad()


class sec:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.cos(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class arcsin:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.arcsin(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class arcos:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.arccos(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class ln:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.log(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class cot:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.tan(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class csc:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.sin(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class arcot:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.arctan(1/val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class tan:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.tan(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class arctan:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.arctan(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __truediv__(self, other):
        return Divide(self, other)


class X:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return val

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)


class Prime:
    def __init__(self, func, grad=1):
        self.grad = grad
        self.result = self.prime(func)
        self.operation = set()

    def prime(self, func):

        if isinstance(func, int or float or cp.ndarray):
            return 0, 0
        if isinstance(func.x, (Divide, Add, Multi, Minus)):
            grad, actual_calculate = func.x.get_grad()
            self.grad *= grad
            func.x = actual_calculate
            func.grad = self.grad
        elif func.x in derivatives.keys():
            prime = Prime(func.x, self.grad)
            grad, actual_calculate = prime.result
            func.x = actual_calculate
            self.grad = prime.grad
            self.grad *= grad
        # if isinstance(func.x, (Divide, Add, Multi, Minus)):
        #     return func.x.get_grad()
        return derivatives[type(func)](func)


all_function = [globals()[name] for name in dir() if callable(globals()[name])]


def get_grads(x, y):
    prime1 = Prime(x)
    prime2 = Prime(y)

    grad1 = prime1.grad * prime1.result[0]
    grad2 = prime2.grad * prime2.result[0]

    return grad1, grad2


def get_grad(x):

    prime1 = Prime(x)
    grad1 = prime1.grad * prime1.result[0]
    return grad1


def get_values(x, y):
    """
    :param x: Any
    :param y: Any
    :return: (x, y)
    """
    if hasattr(x, 'result'):
        x = x.result()
    if hasattr(y, 'result'):
        y = y.result()
    return x, y


def get_value(x):
    """
    :param x: Any
    :return: x
    """
    if hasattr(x, 'result'):
        x = x.result()
    return x


derivatives = {
        expe: lambda func: (func.result(), func.result()),
        sin: lambda func: (cos(func.x).result(), func.result()),
        tan: lambda func: (cp.square(sec(func.x).result()), func.result()),
        sec: lambda func: (func.result()*tan(func.x).result(), func.result()),
        exp: lambda func: (cp.multiply(func.result(), ln(func.val).result()), func.result()),
        ln: lambda func: (1/func.x, func.result()),
        arcsin: lambda func: (1/cp.sqrt(1-cp.square(func.x)), func.result()),
        arcos: lambda func: (-1/cp.sqrt(1 - cp.square(func.x)), func.result()),
        arcot: lambda func: (-1/(1 + cp.square(func.x)), func.result()),
        arctan: lambda func: (1/(1 + cp.square(func.x)), func.result()),
        cos: lambda func: (-sin(func.x).result(), func.result()),
        csc: lambda func: (-func.result()*cot(func.x).result(), func.result()),
        cot: lambda func: (-cp.square(csc(func.x).result()), func.result()),
        Divide: lambda func: func.get_grad(),
        Multi: lambda func: func.get_grad(),
        X: lambda func: (1, func.result()),
    }

print("导数：", get_grad(expe(3)/expe(3)))
print("导数：", get_grad(sin(cot(3)/cot(2*X(3)))))
print("导数：", get_grad(sin(Divide(cot(3), cot(Multi(2, X(3)))))))
