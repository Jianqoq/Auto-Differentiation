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
        result1, result2 = get_values(self.y, self.x)
        return grad1*result1 + result2*grad2


class Divide:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self):
        grad1, grad2 = get_grads(self.x, self.y)
        result1, result2 = get_values(self.y, self.x)
        return (grad1*result1 - result2*grad2)/cp.square(result1)


class expe:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.exp(val)

    def __add__(self, other):
        return self.result + other.result


class exp:
    def __init__(self, x, power):
        self.x = x
        self.power = power

    def result(self):
        val = get_value(self.x)
        return cp.power(val, self.power)

    def __add__(self, other):
        return self.result + other.result


class sin:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.sin(val)


class cos:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.cos(val)


class sec:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.cos(val)

    # def __mul__(self, other):
    #     return self.result * other.result


class arcsin:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.arcsin(val)


class arcos:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.arccos(val)


class ln:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.log(val)

    # def __mul__(self, other):
    #     try:
    #         return self.result()*other.result()
    #     except:
    #         return self.result() * other


class cot:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.tan(val)


class csc:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.sin(val)


class arcot:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.arctan(1/val)


class tan:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.tan(val)

    # def __mul__(self, other):
    #     return self.result * other.result


class arctan:
    def __init__(self, x):
        self.x = x

    def result(self):
        val = get_value(self.x)
        return cp.arctan(val)


class Prime:
    def __init__(self, func, grad=1):
        self.grad = grad
        self.result = self.prime(func)
        self.operation = set()

    def prime(self, func):
        if isinstance(func, int or float or cp.ndarray):
            return 0, 0
        if any(isinstance(func.x, cls) for cls in all_function):
            prime = Prime(func.x, self.grad)
            grad, actual_calculate = prime.result
            func.x = actual_calculate
            self.grad = prime.grad
            self.grad *= grad
        return derivatives[type(func)](func)
        # if isinstance(func, expe):
        #     return func.result(), func.result()
        # elif isinstance(func, sin):
        #     return cos(func.x).result(), func.result()
        # elif isinstance(func, tan):
        #     return cp.square(sec(func.x).result()), func.result()
        # elif isinstance(func, sec):
        #     return func.result()*tan(func.x).result(), func.result()
        # elif isinstance(func, exp):
        #     return cp.multiply(func.result(), cp.log(func.x)), func.result()
        # elif isinstance(func, ln):
        #     return 1/func.x, func.result()
        # elif isinstance(func, arcsin):
        #     return 1/cp.sqrt(1-cp.square(func.x)), func.result()
        # elif isinstance(func, arcos):
        #     return -1/cp.sqrt(1 - cp.square(func.x)), func.result()
        # elif isinstance(func, arcot):
        #     return -1/(1 + cp.square(func.x)), func.result()
        # elif isinstance(func, arctan):
        #     return 1/(1 + cp.square(func.x)), func.result()
        # elif isinstance(func, cos):
        #     return -sin(func.x).result(), func.result()
        # elif isinstance(func, csc):
        #     return -func.result()*cot(func.x).result(), func.result()
        # elif isinstance(func, cot):
        #     return -cp.square(csc(func.x).result()), func.result()


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
        exp: lambda func: (cp.multiply(func.result(), cp.log(func.x)), func.result()),
        ln: lambda func: (1/func.x, func.result()),
        arcsin: lambda func: (1/cp.sqrt(1-cp.square(func.x)), func.result()),
        arcos: lambda func: (-1/cp.sqrt(1 - cp.square(func.x)), func.result()),
        arcot: lambda func: (-1/(1 + cp.square(func.x)), func.result()),
        arctan: lambda func: (1/(1 + cp.square(func.x)), func.result()),
        cos: lambda func: (-sin(func.x).result(), func.result()),
        csc: lambda func: (-func.result()*cot(func.x).result(), func.result()),
        cot: lambda func: (-cp.square(csc(func.x).result()), func.result())
    }
all_function = [globals()[name] for name in dir() if callable(globals()[name])]

print(get_grad(cot(ln(sin(3)))))
