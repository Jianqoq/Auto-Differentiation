import numpy as np
from torch import tensor
import torch
import math

import grad


class Function:
    def __init__(self, x=None, y=None):
        self.grad = None
        self.x = x
        self.y = y
        self._x_val = None
        self._y_val = None
        self.val = self.result()
        self._expression = np.array2string(self.val, prefix=f"{self.__class__.__name__}(", separator=', ')
        self.shape = ()
        if hasattr(self.val, "shape"):
            self.shape = self.val.shape

    def result(self) -> np.ndarray:
        """
        require to implement by yourself
        """
        pass

    def get_grad(self, grad):
        """
        require to implement by yourself
        """
        pass

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False) -> None:
        if grad is None:
            grad = np.array(1.)
        elif isinstance(grad, (list, float)):
            grad = np.array(grad, dtype='float32')
        elif isinstance(grad, np.ndarray):
            pass
            # assert np.issubdtype(grad.dtype, np.integer), "can't use int type"
        else:
            raise TypeError("Only supports float, list and ndarray")
        Prime(self, grad=grad, debug=debug)
        set_grads(self.x, self.y)

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
        return f"{self.__class__.__name__}({self._expression}, shape={self.shape})"

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
        return iter(self.val)

    def __abs__(self):
        return Abs(self)

    def __min__(self):
        return Min(self)


class Matrix(Function):
    def __init__(self, data):
        x = np.array(data)
        super().__init__(x)
        self.x = x
        self.ls = data
        self.shape = x.shape
        self.T = np.transpose(x)
        self.label = None
        self.size = x.size

    def result(self):
        return self.x

    def repeat(self, *direction):
        return repeat(self, direction)

    def __mul__(self, other):
        return starmulti(self, other)

    def __rmul__(self, other):
        return starmulti(other, self)

    def __getitem__(self, item):
        x = self.x[item]
        return Slice(x, self, item)


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
        self.x.grad = Matrix(self.x.grad)
        self.y.grad = Matrix(self.y.grad)

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
        self.x.grad = Matrix(self.x.grad)
        self.y.grad = Matrix(self.y.grad)

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
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def result(self):
        val1, val2 = get_values(self.x, self.y)
        return val1@val2

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
        self.x.grad = Matrix(self.x.grad)
        self.y.grad = Matrix(self.y.grad)


class starmulti(Function):
    def __init__(self, x: Matrix, y: Matrix):
        super().__init__(x, y)
        self.x = x
        self.y = y

    def get_grad(self, grad):
        val1, val2 = get_values(self.x, self.y)
        grad1 = val2*grad
        grad2 = grad*val1
        expression1, expression2 = f"{self.x}*{grad}", f"{grad}*{self.y}"
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def result(self):
        val1, val2 = get_values(self.x, self.y)
        return val1*val2


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
        self.x.grad = Matrix(self.x.grad)


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


class inv(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        val = get_value(self.x)
        assert val.shape == grad.shape
        temp = np.linalg.inv(val)
        grad = np.transpose(temp@np.transpose(grad)@(-temp))  # transpose??
        return check_shape(grad, val)

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
        self.x.grad = Matrix(self.x.grad)


class Add(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = grad, grad, f"{grad}*1", f"{grad}*1"
        self.grad = (grad1, grad2)
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

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
        return check_shape(grad, self.x)

    def result(self):
        result1 = get_value(self.x)
        return np.sum(result1, axis=self.axis)

    def gradient(self, grad=1.0, debug=False):
        Prime(self, grad=grad, debug=debug)
        self.x.grad = Matrix(self.x.grad)

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
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        self._x_val = result1
        self._y_val = result2
        return np.subtract(result1, result2)


class repeat(Function):
    def __init__(self, x, direction: tuple):
        assert 0 not in direction, "repeat has to larger than 0"
        self.batch = direction
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        grad = np.array(grad)
        assert grad.shape == self.val.shape, f"grad shape {grad.shape} != self.val shape {self.val.shape}"
        shape = grad.shape
        for i in range(-1, -len(shape)-1, -1):
            new_shape = change_shape(list(grad.shape), self.x.shape[i], i)
            grad = grad.reshape(new_shape)
            grad = np.sum(grad, axis=len(new_shape)-1+i)
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = np.tile(val, self.batch)
        return self._x_val


class Max(Function):
    def __init__(self, x, axis: int | None = None):
        self.axis = axis
        super().__init__(x)
        self.x = x
        self.label = None

    def get_grad(self, grad):
        mask = (self.x.x == self.val)
        div = mask.sum(axis=self.axis)
        new = mask/div
        return check_shape(grad*new, self.x)

    def result(self):
        result = get_value(self.x)
        if self.axis is not None:
            return np.amax(result, axis=self.axis)
        else:
            return np.max(result)


class Min(Function):
    def __init__(self, x, axis: int | None = None):
        self.axis = axis
        super().__init__(x)
        self.x = x
        self.label = None

    def get_grad(self, grad):
        mask = (self.x.x == self.val)
        div = mask.sum(axis=self.axis)
        new = mask/div
        return check_shape(grad*new, self.x)

    def result(self):
        result = get_value(self.x)
        if self.axis is not None:
            return np.amax(result, axis=self.axis)
        else:
            return np.max(result)


class Multi(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = grad*self._y_val, grad*self._x_val, f"{grad}*{self.y}", f"{grad}*{self.x}"
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        self._x_val = result1
        self._y_val = result2
        return np.multiply(result1, result2)


class mean(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        grad = grad/np.size(self.x.x)
        grad = np.full(self.x.x.shape, grad)
        return grad

    def result(self):
        val = get_value(self.x)
        return np.mean(val)


class Divide(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        self.grad = None

    def get_grad(self, grad):
        grad1 = np.array(grad)/self._y_val
        grad2 = np.array(grad)*(-self._x_val/np.square(self._y_val))
        expression1, expression2 = f"{grad}*(-1)", f"{grad}*1"
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        self._x_val = result1
        self._y_val = result2
        return result1/result2


class exp(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        grad = grad*np.exp(self._x_val)
        return check_shape(grad, self.x)

    def result(self) -> np.ndarray:
        val = get_value(self.x)
        self._x_val = val
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
        grad = grad*self.power*np.power(self._x_val, self.power-1)
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.power(val, self.power)

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
        grad = grad*np.cos(self._x_val)
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.sin(val)


class cos(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        grad = -grad*np.sin(self._x_val)
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.cos(val)


class sec(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        grad = grad*np.tan(self._x_val)*np.sec(self._x_val)
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return 1/np.cos(val)


class arcsin(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        grad = grad*(1/np.sqrt(1-np.square(self._x_val)))
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.arcsin(val)


class arcos(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        grad = grad*(-1/np.sqrt(self._x_val+1))
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.arccos(val)


class ln(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        grad = grad*(1/self._x_val)
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        assert np.all(val > 0), "input has element <= zero"
        self._x_val = val
        return np.log(val)


class log(Function):
    """
    any base log
    """
    def __init__(self, base, x):
        self.base = base
        super().__init__(x)
        self.x = x
        assert base != 0, "base can't be zero"

    def get_grad(self, grad):
        grad = grad*(1/(self._x_val*np.log(self.base)))
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        assert np.all(val > 0), "input has element <= zero"
        self._x_val = val
        return np.log(val)/np.log(self.base)


class cot(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        self.expression = x

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.square(csc(val))
        return check_shape(grad, self.x)

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
        return check_shape(grad, self.x)

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
        return check_shape(grad, self.x)

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
        return check_shape(grad, self.x)

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
        return check_shape(grad, self.x)

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
        return check_shape(grad, self.x)

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
        return check_shape(grad, self.x)

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
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        return np.sqrt(val)


class argmax(Function):
    def __init__(self, x):
        super(argmax, self).__init__(x)
        self.x = x

    def result(self):
        val = get_value(self.x)
        return np.argmax(val)

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False) -> None:
        raise RuntimeError("Doesn't need gradient method")


class argmin(Function):
    def __init__(self, x):
        super(argmin, self).__init__(x)
        self.x = x

    def result(self):
        val = get_value(self.x)
        return np.argmin(val)

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False) -> None:
        raise RuntimeError("Doesn't need gradient method")


class Abs(Function):
    def __init__(self, x):
        super(Abs, self).__init__(x)
        self.x = x

    def result(self):
        val = get_value(self.x)
        return np.abs(val)

    def get_grad(self, grad):
        val = (self.x.x != self.val)
        val = np.where(val == 0, 1, -1)
        return check_shape(grad*val, self.x)


class Prime:
    def __init__(self, func, grad, label="x", express="", debug=False):
        self.grad = grad
        self.express = express
        self.label = label
        self.debug = debug
        self.prime(func)

    def prime(self, func):
        if isinstance(func, (int, float, np.ndarray)):
            if self.debug:
                func._expression = np.array2string(func, prefix=f"Branch 1:  {self.__class__.__name__}(")
                print("Branch 1:", func, id(func))
            return 0, 0, "0"
        elif isinstance(func, (X, Matrix)):
            if self.debug:
                func._expression = np.array2string(func.val, prefix=f"Branch 2:  {self.__class__.__name__}(")
                print("Branch 2:", func, id(func))
            if func.grad is None:
                func.grad = self.grad
            else:
                func.grad += self.grad
        elif isinstance(func, (Divide, Add, Multi, Sub, Matmul, starmulti)):
            if self.debug:
                func._expression = np.array2string(func.val, prefix=f"Branch 3:  {self.__class__.__name__}(")
                print("Branch 3:", func, id(func))
            grad1, grad2, expression1, expression2 = func.get_grad(self.grad)
            Prime(func.x, grad1, debug=self.debug)
            Prime(func.y, grad2, debug=self.debug)
            self.express = (expression1, expression2)
        elif type(func) in derivatives:
            if self.debug:
                func._expression = np.array2string(func.val, prefix=f"Branch 4:  {self.__class__.__name__}(")
                print("Branch 4:", func, id(func))
            grad = func.get_grad(self.grad)
            Prime(func.x, grad)
        elif isinstance(func.x, (int, float, np.ndarray)):
            if self.debug:
                func._expression = np.array2string(func, prefix=f"Branch 5:  {self.__class__.__name__}(")
                print("Branch 5:", func, id(func))
            return 0, 0, "0"


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


def set_grads(x, y):
    """
    :param x: Any
    :return: x
    """
    if hasattr(x, 'grad'):
        x.grad = Matrix(x.grad)
    if hasattr(y, 'grad'):
        y.grad = Matrix(y.grad)


def get_values(x, y):
    """
    :param x: Any
    :param y: Any
    :return: (x, y)
    """
    if hasattr(x, 'result'):
        x = x.val
    if hasattr(y, 'result'):
        y = y.val
    return x, y


def get_value(x):
    """
    :param x: Any
    :return: x
    """
    if hasattr(x, 'result'):
        x = x.val
    return x


def check_shape(grad, inp):
    """
    :return: real grad
    """
    if hasattr(inp, "shape"):
        shape = inp.shape
    else:
        shape = ()
    offset = len(grad.shape) - len(shape)
    if offset < 0:
        raise RuntimeError(f"grad shape {grad.shape} smaller than {shape}")
    for _ in range(offset):
        grad = np.sum(grad, axis=0)
    return grad


def change_shape(x, colms, index):
    temp = x[index]
    x[index] = temp//colms
    x.insert(len(x) + 1 + index, colms)
    return x


class Result:
    def __init__(self, func):
        self.result = self.get_result(func)

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
               X, sqrt, square, transpose, trace, inv, Sum, Max, Slice, reshape, Abs, Min, mean, log, repeat)


if __name__ == "__main__":
    b = tensor(np.arange(0., 120.).reshape((2, 2, 5, 6)), requires_grad=True)
    c = tensor([[7., 8., 9.]], requires_grad=True)
    p = b.repeat(2, 2, 2, 2)
    p.backward(tensor(np.arange(0., 1920.).reshape((4, 4, 10, 12))))
    q = Matrix(np.arange(0., 120.).reshape((2, 2, 5, 6)))
    w = Matrix([[7, 8, 9]])
    i = q.repeat(2, 2, 2, 2)
    o = np.arange(0., 1920.).reshape((4, 4, 10, 12))
    i.gradient(o)
    print(q.grad)
    print(b.grad)


