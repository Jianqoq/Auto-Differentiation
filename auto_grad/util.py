import numpy as np
from sympy import simplify
import ast
from torch import tensor
import torch


class Function:
    def __init__(self):
        self.grad = None
        self.x = None

    def result(self):
        """
        require to implement by yourself
        """
        pass

    def get_grad(self, grad):
        """
        require to implement by yourself
        """
        pass

    def gradient(self, debug=False):
        Prime(self, debug=debug)

    def __str__(self):
        pass

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

    def __matmul__(self, other):
        return Matmul(self, other)

    def __rmatmul__(self, other):
        return Matmul(other, self)


class Matrix(Function):
    """
    size: number of variables. [x1, x2, x3, x4 ... xn]
    label: variable identity. "x" = x vector. "b" = b vector
    """
    def __init__(self, data, label='x'):
        super().__init__()
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

    def __str__(self):
        return self.label


class Matmul(Function):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def get_grad(self, grad):
        grad = grad*self.y.T
        self.grad = grad
        return grad

    def result(self):
        return self.x@self.y


class Add:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.grad = 0

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = grad, grad, f"{grad}*1", f"{grad}*1"
        self.grad = (grad1, grad2)
        return grad1, grad2, expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return np.add(result1, result2)

    def __str__(self):
        return f"({str(self.x)}+{str(self.y)})"


class Sub(Function):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.label = None

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = -1*grad, grad, f"{grad}*(-1)", f"{grad}*1"
        self.grad = (grad1, grad2)
        return grad1, grad2, expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return np.subtract(result1, result2)

    def __str__(self):
        return f"{str(self.x)}-{str(self.y)}"


class Multi(Function):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.label = None

    def get_grad(self, grad):
        val1, val2 = get_values(self.x, self.y)
        grad1, grad2, expression1, expression2 = grad*val2, grad*val1, f"{grad}*{self.y}", f"{grad}*{self.x}"
        return grad1, grad2, expression1, expression2

    def result(self):
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


class Divide(Function):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.grad = None

    def get_grad(self, grad):
        val1, val2 = get_values(self.x, self.y)
        grad1, grad2 = val2*grad, -val1/np.square(val2)*grad
        expression1, expression2 = f"{grad}*(-1)", f"{grad}*1"
        self.grad = (grad1, grad2)
        return grad1, grad2, expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return result1/result2

    def __str__(self):
        return f"{str(self.x)}/{str(self.y)}"


class exp(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.exp(val)
        return grad

    def result(self):
        val = get_value(self.x)
        return np.exp(val)

    def __str__(self):
        return f"exp({str(self.x)})"


class pow(Function):
    def __init__(self, x, power):
        super().__init__()
        self.x = x
        self.power = power
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*self.power*np.power(val, self.power-1)
        return grad

    def result(self):
        return np.power(get_value(self.x), self.power)

    def __str__(self):
        return f"pow({str(self.expression)}, {self.power})"


class sin(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.cos(val)
        return grad

    def result(self):
        val = get_value(self.x)
        return np.sin(val)

    def __str__(self):
        return f"sin({str(self.expression)})"


class cos(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = -grad*np.sin(val)
        return grad

    def result(self):
        val = get_value(self.x)
        return np.cos(val)

    def __str__(self):
        return f"cos({str(self.expression)})"


class sec(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.tan(val)*np.sec(val)
        return grad

    def result(self):
        val = get_value(self.x)
        return 1/np.cos(val)

    def __str__(self):
        return f"sec({str(self.expression)})"


class arcsin(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(1/np.sqrt(1-np.square(val)))
        return grad

    def result(self):
        val = get_value(self.x)
        return np.arcsin(val)

    def __str__(self):
        return f"arcsin({str(self.expression)})"


class arcos(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(-1/np.sqrt(val+1))
        return grad

    def result(self):
        val = get_value(self.x)
        return np.arccos(val)

    def __str__(self):
        return f"arcos({str(self.expression)})"


class ln(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(1/val)
        return grad

    def result(self):
        val = get_value(self.x)
        return np.log(val)

    def __str__(self):
        return f"ln({str(self.expression)})"


class cot(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.square(csc(val))
        return grad

    def result(self):
        val = get_value(self.x)
        return 1/np.tan(val)

    def __str__(self):
        return f"cot({str(self.expression)})"


class csc(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(-1)*np.csc(val)*(cot(val).result())
        return grad

    def result(self):
        val = get_value(self.x)
        return 1/np.sin(val)

    def __str__(self):
        return f"csc({str(self.expression)})"


class arcot(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(-1/(1 + np.square(val)))
        return grad

    def result(self):
        val = get_value(self.x)
        return np.arctan(1/val)

    def __str__(self):
        return f"arcot({str(self.expression)})"


class tan(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(1/np.square(sec(val).result()))
        return grad

    def result(self):
        val = get_value(self.x)
        return np.tan(val)

    def __str__(self):
        return f"tan({str(self.expression)})"


class arctan(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(1/(1 + np.square(val)))
        return grad

    def result(self):
        val = get_value(self.x)
        return np.arctan(val)

    def __str__(self):
        return f"arctan({str(self.expression)})"


class X(Function):
    def __init__(self, x):
        super().__init__()
        self.x = np.array(x)
        self.expression = x
        self.grad = 0

    def get_grad(self, grad):
        return 1

    def result(self):
        val = get_value(self.x)
        return val

    def __str__(self):
        return "x"


class square(Function):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.expression = x
        self.grad = None

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*2*val
        return grad

    def result(self):
        val = get_value(self.x)
        return np.square(val)

    def __str__(self):
        return f"square({str(self.expression)})"


class sqrt:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*0.5*np.power(val, -0.5)
        return grad

    def result(self):
        val = get_value(self.x)
        return np.sqrt(val)

    def __str__(self):
        return f"sqrt({str(self.expression)})"


class Prime:
    def __init__(self, func, grad=1.0, label="x", express="", debug=False):
        self.grad = grad
        self.express = express
        self.label = label
        self.debug = debug
        if debug:
            print("找到可求导变量:", func)
        self.result = self.prime(func)

    def prime(self, func):
        if isinstance(func, (int, float, np.ndarray)):
            if self.debug:
                print("Branch 1(int, float, np.ndarray):", func)
            return 0, 0, "0"
        elif isinstance(func, X):
            if self.debug:
                print("Branch 2(X):", func)
            func.grad += self.grad
        elif isinstance(func, (Divide, Add, Multi, Sub)):
            if self.debug:
                print("Branch 3(Divide, Add, Multi, Sub):", func)
            grad1, grad2, expression1, expression2 = func.get_grad(self.grad)
            Prime(func.x, grad1)
            Prime(func.y, grad2)
            self.express = (expression1, expression2)
        elif type(func) in derivatives:
            if self.debug:
                print("Branch 4(derivatives):", func)
            grad = func.get_grad(self.grad)
            Prime(func.x, grad)
        elif isinstance(func.x, (int, float, np.ndarray)):
            if self.debug:
                print("Branch 5(int, float, np.ndarray):", func)
            return 0, 0, "0"


def get_grads(x, y, label, express="", debug=False):
    prime1 = Prime(x, label=label, express=express, debug=debug)
    prime2 = Prime(y, label=label, express=express, debug=debug)
    grad1 = prime1.grad * prime1.result[0]
    grad2 = prime2.grad * prime2.result[0]
    return grad1, grad2, prime1.result[2], prime2.result[2]


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
        elif type(func.x) in derivatives:
            prime = Result(func.x)
            actual_calculate = prime.result
            func.x = actual_calculate
        return func.result()


derivatives = (exp, sin, tan, sec, pow, ln, arcsin, arcos, arcot, arctan, cos, csc, cot, Divide, Multi, Add, Sub,
               X, sqrt, square, Matmul)


if __name__ == "__main__":
    # a = tensor(1., requires_grad=True)
    # b = tensor([[1.], [2.], [3.]], requires_grad=True)
    # y = a@b
    # y.backward()
    # print(a.grad)
    # w = Matrix(1.)
    # q = Matrix([[1.], [2.], [3.]])
    p = X(3)
    o = X(1)
    po = sin(p*o)
    po.gradient()
    print(p.grad, o.grad)

