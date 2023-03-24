import numpy as np
from sympy import simplify
import ast
from torch import tensor
import torch


class Function:
    def __init__(self, x=None, y=None):
        self.grad = None
        self.x = x
        self.y = y
        self.val = self.result()
        self.shape = 0
        if hasattr(self.val, "shape"):
            self.shape = self.val.shape

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

    def gradient(self, grad:  list | float | np.ndarray, debug=False):
        Prime(self, grad=grad,  debug=debug)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        elif len(shape) == 2 and isinstance(shape[0], int) and isinstance(shape[1], int):
            pass
        else:
            raise TypeError("Invalid arguments, should be either tuple (1, 2) or 1, 2")

        return reshape(self, shape)

    def sum(self):
        pass

    def __str__(self):
        return f"Matrix({self.val}, shape={self.shape})"

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

    def __iter__(self):
        return self.x


class Matrix(Function):
    def __init__(self, data):
        x = np.array(data)
        super().__init__(x)
        self.x = x
        self.ls = data
        self.shape = x.shape
        self.T = np.transpose(x)
        self.size = x.size

    def result(self):
        return self.x

    def __mul__(self, other):
        return elewiseproduct(self, other)

    def __rmul__(self, other):
        return elewiseproduct(other, self)

    def __getitem__(self, item):
        x = self.x[item]
        return Slice(x, self, item)

    def __str__(self):
        return f"Matrix({self.x}, shape={self.shape})"


class Slice(Function):
    def __init__(self, data, origin_data, index):
        super().__init__(origin_data)
        self.x = origin_data
        self.sliced_data = data
        self._shape = data.shape
        self._index = index

    def get_grad(self, grad=None):
        """
        :param grad: same shape as the sliced array shape
        :return: ndarray
        """
        assert grad.shape == self._shape, f"grad shape {grad.shape} doesn't match {self._shape}"
        zeros = np.zeros(self.x.shape)
        zeros[self._index] = grad
        return zeros

    def gradient(self, grad=None, debug=False):
        if grad is None:
            grad = np.ones(self._shape)
        Prime(self, grad=grad, debug=debug)

    def result(self):
        return get_value(self.x)

    def __str__(self):
        return f"Matrix({self.x})"


class reshape(Function):
    def __init__(self, data, shape):
        super().__init__(data)
        self.x = data
        val = get_value(data)
        self.saved_reshape = val.reshape(shape)
        self._shape = val.shape
        self.size = val.size

    def get_grad(self, grad=None):
        assert grad.size == self.size
        grad = grad.reshape(self._shape)
        return grad

    def gradient(self, grad=None, debug=False):
        if grad is None:
            grad = np.ones(self._shape)
        Prime(self, grad=grad, debug=debug)

    def result(self):
        return get_value(self.saved_reshape)

    def __str__(self):
        return f"reshape({self.saved_reshape}, shape={self.saved_reshape.shape})"


class Matmul(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y

    def get_grad(self, grad):
        grad1 = grad@self.y.T
        grad2 = self.x.T@grad
        expression1, expression2 = f"{grad}@{self.y.T}", f"{self.x.T}@{grad}"
        return check_shape(grad1, self.x.shape), check_shape(grad2, self.y.shape), expression1, expression2

    def result(self):
        return self.x@self.y

    def gradient(self, grad: list | None = None, debug=False):
        """
        :param grad: d(loss)/d(self)
        :return: (d(loss)/d(self))@self.y.T, self.x.T@(d(loss)/d(self))
        """
        assert isinstance(grad, list), "Provide list to calculate the grad"
        grad = np.array(grad)
        shape1 = self.x.T.shape
        shape = grad.shape
        shape2 = self.y.T.shape
        assert shape1[1] == shape[0], f"Matrix1 {shape1} != {shape} param"
        assert shape2[0] == shape[1], f"Matrix2 {shape2} != {shape} param"
        Prime(self, grad=grad, debug=debug)


class elewiseproduct(Function):
    def __init__(self, x: Matrix, y: Matrix):
        super().__init__(x, y)
        self.x = x
        self.y = y

    def get_grad(self, grad):
        val1, val2 = get_values(self.x, self.y)
        grad1 = val2*grad
        grad2 = grad*val1
        expression1, expression2 = f"{self.x}*{grad}", f"{grad}*{self.y}"
        return check_shape(grad1, self.x.shape), check_shape(grad2, self.y.shape), expression1, expression2

    def result(self):
        val1, val2 = get_values(self.x, self.y)
        return val1*val2

    def gradient(self, grad: list | None = None, debug=False):
        """
        :param grad: element wise product.
        :return:
        """
        assert isinstance(grad, list), "Provide list to calculate the grad"
        grad = np.array(grad)
        Prime(self, grad=grad, debug=debug)


class transpose(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return val.T

    def gradient(self, grad: list | None = None, debug=False):
        """
        grad has to have the same shape as self.x

        grad: trace((d(loss)/d(self)).T*d(self.x.T))  --> trace(d(loss)/d(self)*d(self.x))

        :return: d(loss)/d(self)
        """
        assert isinstance(grad, list), "Provide list to calculate the grad"
        grad = np.array(grad)
        shape1 = self.x.T.shape
        shape = grad.shape
        assert shape1 == shape, f"Matrix1 {shape1} != {shape} param"
        Prime(self, grad=grad, debug=debug)


class trace(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        val = get_value(self.x)
        assert val.shape[-2] == val.shape[-1], "input has to be square matrix"
        grad = grad*np.identity(val.shape[0])
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.trace(val)

    def gradient(self, grad=1.0, debug=False):
        """
        grad has to have the same shape as self.x

        grad: trace((d(loss)/d(self)).T*d(self.x.T))  --> trace(d(loss)/d(self)*d(self.x))

        :return: d(loss)/d(self)
        """
        assert isinstance(grad, (float, np.ndarray)), "Provide list to calculate the grad"
        Prime(self, grad=grad, debug=debug)


class inv(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        val = get_value(self.x)
        assert val.shape == grad.shape
        temp = np.linalg.inv(val)
        grad = np.transpose(temp@np.transpose(grad)@(-temp))  # transpose??
        return check_shape(grad, val.shape)

    def result(self):
        val = get_value(self.x)
        return np.linalg.inv(val)

    def gradient(self, grad=1.0, debug=False):
        """
        grad has to have the same shape as self.x

        grad: trace((d(loss)/d(self)).T*d(self.x.T))  --> trace(d(loss)/d(self)*d(self.x))

        :return: d(loss)/d(self)
        """
        assert isinstance(grad, list), "Provide list to calculate the grad"
        grad = np.array(grad)
        Prime(self, grad=grad, debug=debug)


class Add(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = grad, grad, f"{grad}*1", f"{grad}*1"
        self.grad = (grad1, grad2)
        return check_shape(grad1, self.x.shape), check_shape(grad2, self.y.shape), expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return np.add(result1, result2)


class Sum(Function):
    def __init__(self, x, axis=0):
        super().__init__(x)
        self.x = x
        self.axis = axis

    def get_grad(self, grad):
        val = get_value(grad)
        grad = np.broadcast_to(val, self.x.x.shape)
        return check_shape(grad, self.x.shape)

    def result(self):
        result1 = get_value(self.x)
        return np.sum(result1, axis=self.axis)

    def gradient(self, grad=1.0, debug=False):
        Prime(self, grad=grad, debug=debug)

    def __str__(self):
        return str(self.result())


class Sub(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.label = None

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = -1*grad, grad, f"{grad}*(-1)", f"{grad}*1"
        return check_shape(grad1, self.x.shape), check_shape(grad2, self.y.shape), expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return np.subtract(result1, result2)


class Max(Function):
    def __init__(self, x, axis=0):
        super().__init__(x)
        self.x = x
        self.axis = axis
        self.label = None

    def get_grad(self, grad):
        mask = (self.x.x == self.val)
        div = mask.sum(axis=self.axis)
        new = mask/div
        return check_shape(grad*new, self.x.shape)

    def result(self):
        result = get_value(self.x)
        return np.amax(result, axis=self.axis)


class Multi(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.label = None

    def get_grad(self, grad):
        val1, val2 = get_values(self.x, self.y)
        grad1, grad2, expression1, expression2 = grad*val2, grad*val1, f"{grad}*{self.y}", f"{grad}*{self.x}"
        return check_shape(grad1, val1.shape), check_shape(grad2, val2.shape), expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return np.multiply(result1, result2)


class Divide(Function):
    def __init__(self, x, y):
        super().__init__(x=x, y=y)
        self.x = x
        self.y = y
        self.grad = None

    def get_grad(self, grad):
        val1, val2 = get_values(self.x, self.y)
        grad1 = np.array(grad)/val2
        grad2 = np.array(grad)*(-val1/np.square(val2))
        expression1, expression2 = f"{grad}*(-1)", f"{grad}*1"
        return check_shape(grad1, val1.shape), check_shape(grad2, val2.shape), expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return result1/result2

    def gradient(self, grad: list | np.ndarray, debug=False):
        Prime(self, grad=grad, debug=debug)


class exp(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.exp(val)
        return check_shape(grad, self.x.shape)

    def result(self) -> np.ndarray:
        val = get_value(self.x)
        return np.exp(val)

    def __getitem__(self, item):
        return self.x.x[item]


class pow(Function):
    def __init__(self, x, power):
        super().__init__(x, power)
        self.x = x
        self.power = power
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*self.power*np.power(val, self.power-1)
        return check_shape(grad, self.x.shape)

    def result(self):
        return np.power(get_value(self.x), self.power)

    def gradient(self, grad=None, debug=False):
        """
        grad has to have the same shape as self.x

        grad: trace((d(loss)/d(self)).T*d(self.x.T))  --> trace(d(loss)/d(self)*d(self.x))

        :return: d(loss)/d(self)
        """
        if grad is None:
            grad = 1.0
        grad = np.array(grad)
        Prime(self, grad=grad, debug=debug)


class sin(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.cos(val)
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.sin(val)


class cos(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = -grad*np.sin(val)
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.cos(val)


class sec(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.tan(val)*np.sec(val)
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return 1/np.cos(val)


class arcsin(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(1/np.sqrt(1-np.square(val)))
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.arcsin(val)


class arcos(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(-1/np.sqrt(val+1))
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.arccos(val)


class ln(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(1/val)
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.log(val)


class cot(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.square(csc(val))
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return 1/np.tan(val)


class csc(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(-1)*np.csc(val)*(cot(val).result())
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return 1/np.sin(val)


class arcot(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(-1/(1 + np.square(val)))
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.arctan(1/val)


class tan(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(1/np.square(sec(val).result()))
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.tan(val)


class arctan(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*(1/(1 + np.square(val)))
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.arctan(val)


class X(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = np.array(x)
        self.expression = x
        self.grad = 0

    def get_grad(self, grad):
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return val


class square(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x
        self.grad = None

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*2*val
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.square(val)


class sqrt:
    def __init__(self, x):
        self.x = x
        self.expression = x
        self.grad = None

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*0.5*np.power(val, -0.5)
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return np.sqrt(val)


class Prime:
    def __init__(self, func, grad=np.array(1.0), label="x", express="", debug=False):
        self.grad = grad
        self.express = express
        self.label = label
        self.debug = debug
        self.prime(func)

    def prime(self, func):
        if isinstance(func, (int, float, np.ndarray)):
            if self.debug:
                print("Branch 1: ", func, id(func))
            return 0, 0, "0"
        elif isinstance(func, (X, Matrix)):
            if self.debug:
                print("Branch 2: ", func, id(func))
            if func.grad is None:
                func.grad = Matrix(self.grad)
            else:
                func.grad += Matrix(self.grad)
        elif isinstance(func, (Divide, Add, Multi, Sub, Matmul, elewiseproduct)):
            if self.debug:
                print("Branch 3: ", type(func))
                print(type(func.x), type(func.y), id(func.x), id(func.y))
            grad1, grad2, expression1, expression2 = func.get_grad(self.grad)
            Prime(func.x, grad1, debug=self.debug)
            Prime(func.y, grad2, debug=self.debug)
            self.express = (expression1, expression2)
        elif type(func) in derivatives:
            if self.debug:
                print("Branch 4: ", type(func), id(func))
            grad = func.get_grad(self.grad)
            Prime(func.x, grad)
        elif isinstance(func.x, (int, float, np.ndarray)):
            if self.debug:
                print("Branch 5:", type(func), id(func))
            return 0, 0, "0"


def get_grads(x, y, label, express="", debug=False):
    prime1 = Prime(x, label=label, express=express, debug=debug)
    prime2 = Prime(y, label=label, express=express, debug=debug)
    grad1 = prime1.grad * prime1.result[0]
    grad2 = prime2.grad * prime2.result[0]
    return grad1, grad2, prime1.result[2], prime2.result[2]


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


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


def check_shape(grad, input_shape):
    """
    :return: real grad
    """
    offset = len(grad.shape) - len(input_shape)
    for _ in range(offset):
        grad = np.sum(grad, axis=0)
    return grad


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


derivatives = (exp, sin, tan, sec, pow, ln, arcsin, arcos, arcot, arctan, cos, csc, cot, Divide, Add, Multi, Sub,
               X, sqrt, square, transpose, trace, inv, Sum, Max, Slice, reshape)


if __name__ == "__main__":
    b = tensor([1., 5., 3.], requires_grad=True)
    c = tensor([[1., 2., 3.], [2., 1., 8.], [6., 2., 3.]], requires_grad=True)
    p = b*c*b
    p.backward(tensor([[1., 5., 3.], [2., 5., 8.], [6., 5., 3.]]))
    print(b.grad)
    q = Matrix([1., 5., 3.])
    w = Matrix([[1., 2., 3.], [2., 1., 8.], [6., 2., 3.]])
    p = q*w*q
    p.gradient([[1., 5., 3.], [2., 5., 8.], [6., 5., 3.]])
    print(q.grad)

