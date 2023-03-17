import cupy as cp
import sympy
from sympy import simplify


class Add:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self):
        grad1, grad2 = get_grads(self.x, self.y)
        val1, val2 = get_values(self.x, self.y)
        return grad1 + grad2, val1+val2

    def __str__(self):
        return f"{str(self.x)}+{str(self.y)}"


class Sub:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self):
        grad1, grad2 = get_grads(self.x, self.y)
        return grad1 - grad2

    def __str__(self):
        return f"{str(self.x)}-{str(self.y)}"


class Multi:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self):
        grad1, grad2, expression1, expression2 = get_grads(self.x, self.y)
        result1, result2 = get_values(self.x, self.y)
        tp = (int, float, cp.ndarray)
        if isinstance(self.x, tp) and self.x < 0:
            if isinstance(self.y, tp) and self.y < 0:
                expression = f"{str(expression1)}*({str(self.y)}) + ({str(self.x)})*{str(expression2)}"
            else:
                expression = f"{str(expression1)}*{str(self.y)} + ({str(self.x)})*{str(expression2)}"
        elif isinstance(self.y, tp) and self.y < 0:
            expression = f"{str(expression1)}*({str(self.y)}) + {str(self.x)}*{str(expression2)}"
        else:
            expression = f"{str(expression1)}*{str(self.y)} + {str(self.x)}*{str(expression2)}"
        return grad1*result2 + result1*grad2, result1*result2, expression

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return result1*result2

    def __str__(self):
        tp = (int, float, cp.ndarray)
        if isinstance(self.x, tp) and self.x < 0:
            if isinstance(self.y, tp) and self.y < 0:
                return f"({str(self.x)})*({str(self.y)})"
            else:
                return f"({str(self.x)})*{str(self.y)}"
        elif isinstance(self.y, tp) and self.y < 0:
            return f"{str(self.x)}*({str(self.y)})"
        else:
            return f"{str(self.x)}*{str(self.y)}"

    def __neg__(self):
        return Multi(-1, self)


class Divide:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_grad(self,):
        grad1, grad2 = get_grads(self.x, self.y)
        result1, result2 = get_values(self.x, self.y)
        return (grad1*result2 - result1*grad2)/cp.square(result2), result1/result2

    def __str__(self):
        return f"{str(self.x)}/{str(self.y)}"


class exp:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.exp(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"exp({str(self.x)})"

    def __neg__(self):
        return Multi(-1, self)


class power:
    def __init__(self, x, pow):
        self.x = x
        self.power = pow
        self.expression = x

    def result(self):
        return cp.power(get_value(self.x), self.power)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"power({str(self.expression)}, {self.power})"

    def __neg__(self):
        return Multi(-1, self)


class sin:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.sin(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __pow__(self, power, modulo=None):
        return

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"sin({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, sin(self.x))


class cos:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.cos(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"cos({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)


class sec:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.cos(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"sec({str(self.x)})"

    def __neg__(self):
        return Multi(-1, self)


class arcsin:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.arcsin(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"arcsin({str(self.x)})"

    def __neg__(self):
        return Multi(-1, self)


class arcos:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.arccos(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"arcos({str(self.x)})"

    def __neg__(self):
        return Multi(-1, self)


class ln:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.log(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"ln({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)


class cot:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.tan(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"cot({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)


class csc:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return 1/cp.sin(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"csc({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)


class arcot:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.arctan(1/val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"arcot({str(self.x)})"

    def __neg__(self):
        return Multi(-1, self)


class tan:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.tan(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"tan({str(self.x)})"

    def __neg__(self):
        return Multi(-1, self)


class arctan:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return cp.arctan(val)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return f"arctan({str(self.x)})"

    def __neg__(self):
        return Multi(-1, self)


class X:
    def __init__(self, x):
        self.x = x
        self.expression = x

    def result(self):
        val = get_value(self.x)
        return val

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __str__(self):
        return 'x'

    def __neg__(self):
        return Multi(-1, self)
        # if type(self.x) not in keys:
        #     return f"-x*({str(self.expression)})"
        # self.x = -self.x
        # return self.result()


class square:
    def __init__(self, x):
        self.expression = x

    def __neg__(self):
        return f"(-square({str(self.expression)}))"

    def __str__(self):
        return f"square({str(self.expression)})"

    def __rsub__(self, other):
        val = get_value(other)
        return f"{val}-square({str(self.expression)})"


class sqrt:
    def __init__(self, x):
        self.expression = x

    def __neg__(self):
        return f"(-sqrt({str(self.expression)}))"

    def __str__(self):
        return f"sqrt({str(self.expression)})"


class Prime:
    def __init__(self, func, grad=1.0, head=True):
        self.grad = grad
        self.express = ""
        print("找到可求导变量:", func)
        self.result = self.prime(func)
        print("求导结果:", self.result[2])

    def prime(self, func):
        if isinstance(func, int or float or cp.ndarray):
            return 0, 0, f"0"
        elif isinstance(func, (Divide, Add, Multi, Sub)):
            grad, actual_calculate, expression = func.get_grad()
            self.grad = grad
            self.express = f"*{expression}"
            return 1, 1, expression
        elif isinstance(func.x, (Divide, Add, Multi, Sub)):
            grad, actual_calculate, expression = func.x.get_grad()
            self.grad *= grad
            func.grad = self.grad
            self.express = f"*{expression}"
        elif type(func.x) in derivatives.keys():
            prime = Prime(func.x, self.grad, False)
            grad, actual_calculate, expression = prime.result
            func.x = actual_calculate
            self.express = f"*{expression}"
            self.grad = prime.grad
            self.grad *= grad
        result = derivatives[type(func)](func)
        express = result[2] + f"{self.express}"
        return result[0], result[1], express


def get_grads(x, y):
    prime1 = Prime(x, head=False)
    prime2 = Prime(y, head=False)
    grad1 = prime1.grad * prime1.result[0]
    grad2 = prime2.grad * prime2.result[0]
    return grad1, grad2, prime1.result[2], prime2.result[2]


def get_grad(x):
    prime1 = Prime(x)
    grad1 = prime1.grad * prime1.result[0]
    return grad1, str(x), prime1.result[2]


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
        exp: lambda func: (cp.exp(func.x), func.result(), str(exp(func.expression))),
        sin: lambda func: (cos(func.x).result(), func.result(), str(cos(func.expression))),
        tan: lambda func: (cp.square(sec(func.x).result()), func.result(),
                           f"1/{str(square(sec(func.expression)))}"),
        sec: lambda func: (func.result()*tan(func.x).result(), func.result(),
                           f"{str(str(sec(func.expression)))}*{str(tan(func.expression))}"),
        power: lambda func: (func.power*(cp.power(func.x, (func.power - 1))), func.result(),
                             str(func.power*(power(func.expression, (func.power - 1))))),
        ln: lambda func: (1/func.x, func.result(), f"1/{str(func.expression)}"),
        arcsin: lambda func: (1/cp.sqrt(1-cp.square(func.x)), func.result(),
                              f"1/{str(sqrt(1 -square(func.expression)))})"),
        arcos: lambda func: (-1/cp.sqrt(1 - cp.square(func.x)), func.result(),
                             f"-1/(1 + {str(sqrt(func.expression))})"),
        arcot: lambda func: (-1/(1 + cp.square(func.x)), func.result(),
                             f"-1/(1 + {str(square(func.expression))})"),
        arctan: lambda func: (1/(1 + cp.square(func.x)), func.result(),
                              f"1/(1 + {str(square(func.expression))})"),
        cos: lambda func: (-sin(func.x).result(), func.result(), str(-sin(func.expression))),
        csc: lambda func: (-func.result()*cot(func.x).result(), func.result(),
                           str(-func*cot(func.expression))),
        cot: lambda func: (-cp.square(csc(func.x).result()), func.result(), str(square(csc(func.expression)))),
        Divide: lambda func: func.get_grad(),
        Multi: lambda func: func.get_grad(),
        Add: lambda func: func.get_grad(),
        Sub: lambda func: func.get_grad(),
        X: lambda func: (1, func.result(), "1"),
    }

keys = list(derivatives.keys())

w = X(-1)
# print(-csc(w))
grad, expression, grad_expression = get_grad(-cos(w))
print(f"导数： {grad},\t表达式： {expression}, \t求导表达式： {grad_expression}\n")
print(f"简化求导表达式： {simplify(grad_expression)}\n")

# 验证
# h2 = cos(csc(power(w, 3))).result()
# h3 = -csc(power(w, 3)).result()
# h4 = cot(power(w, 3)).result()
# h5 = 3*power(w, 2).result()
# h6 = cos(w).result()
# e1 = sin(csc(power(w, 3))).result()
# e2 = -sin(w).result()
# print("导数：", h2*h3*h4*h5*h6 + e1*e2)

# sympy.solve(-3*power(w, 2)*cos(w)*cos(cos(w)*csc(power(w, 3)))*cot(power(w, 3))*csc(power(w, 3)))
