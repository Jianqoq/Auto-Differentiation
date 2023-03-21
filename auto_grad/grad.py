import numpy as np
from sympy import simplify
import ast
from torch import tensor


class Matrix:
    """
    size: number of variables. [x1, x2, x3, x4 ... xn]
    label: variable identity. "x" = x vector. "b" = b vector
    """
    def __init__(self, data: np.ndarray, label='x'):
        self.x = np.array(data)
        self.expression = label
        self.label = label
        self.T = np.transpose(self.x)

    def result(self):
        return self.x

    def __iter__(self):
        return self.x

    def __getitem__(self, item):
        return self.x[item]

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Matmul(self, other)

    def __rmul__(self, other):
        return Matmul(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)

    def __str__(self):
        return self.label

    def __matmul__(self, other):
        result1, result2 = get_values(self, other)
        return result1@result2


class Matmul:
    def __init__(self, x: Matrix, y: Matrix):
        self.x = x
        self.y = y
        self.label = None
        self.grad = None

    def get_grad(self, express="", debug=False):
        grad = self.y.T
        # grad1, grad2, expression1, expression2 = get_grads(self.x, self.y, label, express=express, debug=debug)
        result1, result2 = get_values(self.x, self.y)
        self.grad = grad
        return grad, result1@result2, f"{str(self.x)}*{str(self.y)}.T"

    def result(self):
        return self.x@self.y


class Add:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.label = None
        self.grad = None

    def get_grad(self, label, express="", debug=False):
        grad1, grad2, expression1, expression2 = get_grads(self.x, self.y, label, express=express, debug=debug)
        result1, result2 = get_values(self.x, self.y)
        self.grad = grad1 + grad2
        return grad1 + grad2, result1 + result2, f"({str(expression1)}) + ({str(expression2)})"

    def result(self):
        # print("Add")
        result1, result2 = get_values(self.x, self.y)
        return np.add(result1, result2)

    def __str__(self):
        return f"({str(self.x)}+{str(self.y)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)


class Sub:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.label = None
        self.grad = None

    def get_grad(self, label, express="", debug=False):
        grad1, grad2, expression1, expression2 = get_grads(self.x, self.y, label, express=express, debug=debug)
        result1, result2 = get_values(self.x, self.y)
        self.grad = grad1 - grad2
        return self.grad, result1 - result2, f"({str(expression1)}) - ({str(expression2)})"

    def result(self):
        # print("Sub")
        result1, result2 = get_values(self.x, self.y)
        return np.subtract(result1, result2)

    def __str__(self):
        return f"{str(self.x)}-{str(self.y)}"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)


class Multi:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.label = None
        self.grad = None

    def get_grad(self, label, express="", debug=False):
        grad1, grad2, expression1, expression2 = get_grads(self.x, self.y, label=label, express=express, debug=debug)
        result1, result2 = get_values(self.x, self.y)
        tp = (int, float, np.ndarray)
        if isinstance(self.x, tp) and self.x < 0:
            if isinstance(self.y, tp) and self.y < 0:
                expression = f"({str(expression1)})*({str(self.y)}) + ({str(self.x)})*({str(expression2)})"
            else:
                expression = f"({str(expression1)})*{str(self.y)} + ({str(self.x)})*({str(expression2)})"
        elif isinstance(self.y, tp) and self.y < 0:
            expression = f"({str(expression1)})*({str(self.y)}) + {str(self.x)}*({str(expression2)})"
        else:
            expression = f"({str(expression1)})*{str(self.y)} + {str(self.x)}*({str(expression2)})"
        self.grad = grad1*result2 + result1*grad2
        return self.grad, result1*result2, expression

    def result(self):
        # print("Multi")
        result1, result2 = get_values(self.x, self.y)
        return np.multiply(result1, result2)

    def __str__(self):
        tp = (int, float, np.ndarray)
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

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __rmul__(self, other):
        return Multi(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)


class Divide:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.label = None
        self.grad = None

    def get_grad(self, label, express="", debug=False):
        grad1, grad2, expression1, expression2 = get_grads(self.x, self.y, label, express=express, debug=debug)
        result1, result2 = get_values(self.x, self.y)
        tp = (int, float, np.ndarray)
        if isinstance(self.x, tp) and self.x < 0:
            if isinstance(self.y, tp) and self.y < 0:
                expression = f"({str(expression1)})*({str(self.y)}) - ({str(self.x)})*({str(expression2)})/square({str(self.y)})"
            else:
                expression = f"({str(expression1)})*{str(self.y)} - ({str(self.x)})*({str(expression2)})/square({str(self.y)})"
        elif isinstance(self.y, tp) and self.y < 0:
            expression = f"({str(expression1)})*({str(self.y)}) - {str(self.x)}*({str(expression2)})/square({str(self.y)})"
        else:
            expression = f"({str(expression1)})*{str(self.y)} - {str(self.x)}*({str(expression2)})/square({str(self.y)})"
        self.grad = (grad1*result2 - result1*grad2)/np.square(result2)
        return self.grad, result1/result2, expression

    def result(self):
        # print("Divide")
        result1, result2 = get_values(self.x, self.y)
        return result1/result2

    def __str__(self):
        return f"{str(self.x)}/{str(self.y)}"

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


class exp:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.exp(val)

    def __str__(self):
        return f"exp({str(self.x)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)


class pow:
    def __init__(self, x, power):
        self.x = x
        self.power = power
        self.expression = x
        self.grad = None

    def result(self):
        return np.power(get_value(self.x), self.power)

    def __str__(self):
        return f"pow({str(self.expression)}, {self.power})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class sin:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.sin(val)

    def __str__(self):
        return f"sin({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class cos:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.cos(val)

    def __str__(self):
        return f"cos({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class sec:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return 1/np.cos(val)

    def __str__(self):
        return f"sec({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class arcsin:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.arcsin(val)

    def __str__(self):
        return f"arcsin({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class arcos:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.arccos(val)

    def __str__(self):
        return f"arcos({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class ln:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.log(val)

    def __str__(self):
        return f"ln({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class cot:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return 1/np.tan(val)

    def __str__(self):
        return f"cot({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class csc:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return 1/np.sin(val)

    def __str__(self):
        return f"csc({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class arcot:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.arctan(1/val)

    def __str__(self):
        return f"arcot({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class tan:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.tan(val)

    def __str__(self):
        return f"tan({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class arctan:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.arctan(val)

    def __str__(self):
        return f"arctan({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class X:
    def __init__(self, x, label="x"):
        self.x = np.array(x)
        self.expression = x
        self.grad = None
        self.label = label

    def result(self):
        val = get_value(self.x)
        return val

    def __str__(self):
        return self.label

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class square:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.square(val)

    def __str__(self):
        return f"square({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class sqrt:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def result(self):
        val = get_value(self.x)
        return np.sqrt(val)

    def __str__(self):
        return f"sqrt({str(self.expression)})"

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)


class Xsequence:
    """
    size: number of variables. [x1, x2, x3, x4 ... xn]
    label: variable identity. "x" = x vector. "b" = b vector
    """
    def __init__(self, size, label='x'):
        self.x = np.arange(0, size).reshape(size, 1)
        self.expression = size
        self.label = label
        self.T = np.transpose(self.x)

    def result(self):
        return self.x

    def __iter__(self):
        return self.x

    def __getitem__(self, item):
        return self.x[item]

    def __neg__(self):
        return Multi(-1, self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Multi(self, other)

    def __rmul__(self, other):
        return Multi(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __pow__(self, p, modulo=None):
        return pow(self, p)

    def __str__(self):
        return self.label


class Prime:
    def __init__(self, func, grad=1.0, label="x", express="", debug=False):
        self.grad = grad
        self.express = express
        self.label = label
        self.debug = debug
        if debug:
            print("找到可求导变量:", func)
        self.result = self.prime(func)
        self.express = self.result[2]
        if debug:
            print("求导结果:", self.result[2])

    def prime(self, func):
        if isinstance(func, (int, float, np.ndarray)):
            return 0, 0, "0"
        elif isinstance(func, Xsequence) and self.label is not None and func.label != self.label:
            self.express = "0"
            func.grad = 0
            return 0, 0, "0"
        elif isinstance(func, X) and self.label is not None and func.label == self.label:
            func.grad = 1
            self.express = "1"
        elif isinstance(func, X) and self.label is not None and func.label != self.label:
            func.grad = 0
            self.express = "0"
        elif isinstance(func, (Divide, Add, Multi, Sub)):
            grad, actual_calculate, expression = func.get_grad(label=self.label, express=self.express, debug=self.debug)
            self.grad = grad
            self.express = expression
            # print(expression)
            return 1, 1, expression
        elif isinstance(func.x, (Divide, Add, Multi, Sub)):
            grad, actual_calculate, expression = func.x.get_grad(label=self.label, express=self.express, debug=self.debug)
            func.x = actual_calculate
            self.grad *= grad
            self.express = f"({expression})"
        elif type(func.x) in derivatives.keys():
            prime = Prime(func.x, self.grad, label=self.label, express=self.express, debug=self.debug)
            grad, actual_calculate, expression = prime.result
            self.grad = prime.grad
            self.grad *= grad
            self.express = f"({expression})"
            func.x = actual_calculate
        elif isinstance(func.x, (int, float, np.ndarray)):
            return 0, 0, "0"
        result = derivatives[type(func)](func, self.express)
        func.grad = result[1]
        return result


def get_grads(x, y, label, express="", debug=False):
    prime1 = Prime(x, label=label, express=express, debug=debug)
    prime2 = Prime(y, label=label, express=express, debug=debug)
    grad1 = prime1.grad * prime1.result[0]
    grad2 = prime2.grad * prime2.result[0]
    return grad1, grad2, prime1.result[2], prime2.result[2]


def get_grad(x, label=None, debug=False):
    if label:
        prime1 = Prime(x, label=label, debug=debug)
    else:
        prime1 = Prime(x, label="", debug=debug)
    grad1 = prime1.grad * prime1.result[0]
    return grad1, str(x), prime1.result[2]


def set_grad(x, grad):
    """
    :param x: Any
    :return: x
    """
    if hasattr(x, 'grad'):
        x.grad = grad


def set_grads(x, y, grad1, grad2):
    """
    :param x: Any
    :return: x
    """
    if hasattr(x, 'grad'):
        x.grad = grad1
    if hasattr(y, 'grad'):
        x.grad = grad2


def get_values(x, y):
    """
    :param x: Any
    :param y: Any
    :return: (x, y)
    """
    # print(f"{type(x).__name__}:", x, "\t", f"{type(y).__name__}:", y)
    if hasattr(x, 'result'):
        x = x.result()
    if hasattr(y, 'result'):
        y = y.result()
    # print(f"{type(x).__name__}:", x, "\t", f"{type(y).__name__}:",  y)
    return x, y


def get_value(x):
    """
    :param x: Any
    :return: x
    """
    if hasattr(x, 'result'):
        x = x.result()
    return x


def run_simplified(func):
    func = f"get_grad({func})"
    tree = ast.parse(func, mode='eval')
    grad3, expression3, grad_expression = eval(compile(tree, '', 'eval'))
    print(f"导数： {grad3},\t表达式： {expression3}, \t求导表达式： {grad_expression}\n")
    print(f"简化求导表达式： {str(simplify(grad_expression))}\n")


def get_result(string):
    try:
        tree = ast.parse(string, mode='eval')
        func = eval(compile(tree, '', 'eval'))
        p = Result(func)
        print("结果:", p.result)
    except NameError as e:
        q = string.replace(e.name, 'r')
        tree = ast.parse(q, mode='eval')
        func = eval(compile(tree, '', 'eval'))
        p = Result(func)
        print("结果:", p.result)


class Result:
    def __init__(self, func, head=True):
        # print("找到可求导变量:", type(func))
        self.result = self.get_result(func)
        # print("结果:", self.result)

    def get_result(self, func):
        if isinstance(func, (int, float, np.ndarray)):
            return func
        elif isinstance(func, (Divide, Add, Multi, Sub)):
            actual_calculate = func.result()
            return actual_calculate
        elif isinstance(func.x, (Divide, Add, Multi, Sub)):
            actual_calculate = func.x.result()
            func.x = actual_calculate
        elif type(func.x) in derivatives.keys():
            prime = Result(func.x)
            actual_calculate = prime.result
            func.x = actual_calculate
        return func.result()


derivatives = {
        exp: lambda func, express: (np.exp(func.x), func.result(), str(exp(func.expression))),
        sin: lambda func, express: (cos(func.x).result(), func.result(), f"({str(cos(func.expression))})*{express}"),
        tan: lambda func, express: (np.square(sec(func.x).result()), func.result(),
                                    f"(1/{str(square(sec(func.expression)))})*{express}"),
        sec: lambda func, express: (func.result()*tan(func.x).result(), func.result(),
                                    f"({str(str(sec(func.expression)))}*{str(tan(func.expression))})*{express}"),
        pow: lambda func, express: (func.power*(np.power(func.x, (func.power - 1))), func.result(),
                                    f"({func.power}*pow({func.expression}, {func.power - 1}))*{express}"),
        ln: lambda func, express: (1/func.x, func.result(), f"(1/{str(func.expression)})*{express}"),
        arcsin: lambda func, express: (1/np.sqrt(1-np.square(func.x)), func.result(),
                                       f"(1/{str(sqrt(1 -square(func.expression)))})*{express}"),
        arcos: lambda func, express: (-1/np.sqrt(1 - np.square(func.x)), func.result(),
                                      f"(-1/{str(sqrt(func.expression + 1))})*{express}"),
        arcot: lambda func, express: (-1/(1 + np.square(func.x)), func.result(),
                                      f"(-1/(1 + {str(square(func.expression))}))*{express}"),
        arctan: lambda func, express: (1/(1 + np.square(func.x)), func.result(),
                                       f"(1/(1 + {str(square(func.expression))}))*{express}"),
        cos: lambda func, express: (-sin(func.x).result(), func.result(), f"{str(-sin(func.expression))}*{express}"),
        csc: lambda func, express: (-func.result()*cot(func.x).result(), func.result(),
                                    f"{str(-func*cot(func.expression))}*{express}"),
        cot: lambda func, express: (-np.square(csc(func.x).result()), func.result(), f"{str(square(csc(func.expression)))}*{express}"),
        Divide: lambda func, express: func.get_grad(),
        Multi: lambda func, express: func.get_grad(),
        Add: lambda func, express: func.get_grad(),
        Sub: lambda func, express: func.get_grad(),
        X: lambda func, express: (func.grad, func.result(), express),
        sqrt: lambda func, express: (1/2*np.sqrt(func.x), func.result(), f"(0.5*{func.expression}**(-0.5))*{express}"),
        square: lambda func, express: (2*func.x, func.result(), f"(2*{func.expression})*{express}"),
        Xsequence: lambda func, express: (1, func.result(), "x")
    }

keys = list(derivatives.keys())


if __name__ == "__main__":
    a = tensor(2., requires_grad=True)
    b = tensor(1., requires_grad=True)
    w = X(2., "w")
    q = X(1., "q")
    y = (a + a*b)*(b + 3)
    y.backward()
    print(a.grad, b.grad)
    kl = w+q
    po = w/q
    grad3, expression3, grad_expression = get_grad(po, "w", debug=True)
    print(f"导数： {grad3}\n表达式： {expression3} \n求导表达式： {grad_expression}\n")
    express = str(simplify(grad_expression))
    print(f"简化求导表达式： {express}")
    get_result(express)

