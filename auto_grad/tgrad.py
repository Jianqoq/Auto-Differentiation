import numpy as np
import graphviz
from sympy import simplify
import random
from time import time


class Function:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        if y is None:
            self.arguments = (x,)
        else:
            self.arguments = (x, y)
        self._x_val = None
        self._y_val = None
        self.val = self.result()
        self._expression = np.array2string(self.val, prefix=f"{self.__class__.__name__}(", separator=', ')
        self.shape = ()
        self.graph = None
        self.holder = 0
        self.parent = None
        self.son = None
        self.view = None
        self.label = None
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

    def get_graph(self):
        return

    def visualize(self, open_file=False, size="12, 12"):
        begin = time()
        total = search_nodes(self)
        p = View(self, head=True, time_begin=begin, total_task=total)
        p.graph_object.graph_attr['size'] = size
        p.graph_object.view() if open_file else p.graph_object

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False, create_graph=False) -> None:
        if grad is None:
            grad = np.array(1.)
        elif isinstance(grad, (list, float)):
            grad = np.array(grad)
        elif isinstance(grad, np.ndarray):
            pass
        else:
            raise TypeError("Only supports float, list and ndarray")
        Prime(self, grad=grad, debug=debug, head=True, create_graph=create_graph)

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
        return Multi(Matrix(-1), self)

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
    def __init__(self, data, label="x"):
        x = np.array(data)
        super().__init__(x)
        self.label = label
        self.grad = None
        self.x = x
        self.ls = data
        self.shape = x.shape
        self.T = np.transpose(x)
        self.size = x.size
        self.graph = None

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

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False, create_graph=False):
        if grad is None:
            grad = np.ones(self._shape)
        Prime(self, grad=grad, debug=debug)
        self.x.grad = Matrix(self.x.grad)
        self.y.grad = Matrix(self.y.grad)

    def result(self):
        return get_value(self.x)


class reshape(Function):
    def __init__(self, data, shape):
        super().__init__(data)
        self.x = data
        val = get_value(data)
        self.saved_reshape = val.reshape(shape)
        self._shape = val.shape
        self.size = val.size

    def get_grad(self, create_graph, grad=None):
        assert grad.size == self.size
        grad = grad.reshape(self._shape)
        return grad

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False, create_graph=False):
        if grad is None:
            grad = np.ones(self._shape)
        Prime(self, grad=grad, debug=debug)
        self.x.grad = Matrix(self.x.grad)
        self.y.grad = Matrix(self.y.grad)

    def result(self):
        return get_value(self.saved_reshape)


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

    def gradient(self, grad: list | None = None, debug=False, create_graph=False):
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
        labels = get_label(x, y)
        self.label = f"({labels[0]} * {labels[1]})"

    def get_grad(self, grad):
        val1, val2 = get_values(self.x, self.y)
        grad1 = val2*grad
        grad2 = grad*val1
        expression1, expression2 = f"{self.x}*{grad}", f"{grad}*{self.y}"
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        graph = (starmulti(self.y, self.holder), starmulti(self.holder, self.x))
        return graph

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

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False, create_graph=False):
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

    def gradient(self, grad=1.0, debug=False, create_graph=False):
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
        labels = get_label(x, y)
        self.label = f"({labels[0]} + {labels[1]})"

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = grad+1e-9, grad+1e-10, grad, grad
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return np.add(result1, result2)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return Multi(self.holder, Matrix(1.)), Multi(Matrix(1.), self.holder)


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

    def gradient(self, grad=1.0, debug=False, create_graph=False):
        Prime(self, grad=grad, debug=debug)
        self.x.grad = Matrix(self.x.grad)


class Sub(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        labels = get_label(x, y)
        self.label = f"({labels[0]} - {labels[1]})"

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = -1*grad, grad, f"{grad}*(-1)", f"{grad}*1"
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return Multi(Matrix(-1.), self.holder), Multi(self.holder, Matrix(1.))

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
        x = self.x
        assert grad.shape == self.val.shape, f"grad shape {grad.shape} != self.val shape {self.val.shape}"
        length = len(grad.shape)
        for i in range(-1, -length-1, -1):
            new_shape = change_shape(list(grad.shape), x.shape[i], i)
            grad = grad.reshape(new_shape)
            grad = np.sum(grad, axis=length+i)
        return check_shape(grad, x)

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
        assert self.x.shape == self.val.shape, f"self.x shape{self.x.shape} != self.val shape {self.val.shape}"
        mask = (self.x.x == self.val)
        div = mask.sum(axis=self.axis)
        new = mask/div
        return check_shape(grad*new, self.x)

    def result(self):
        result = get_value(self.x)
        return np.amax(result, axis=self.axis) if self.axis is not None else np.max(result)


class Min(Function):
    def __init__(self, x, axis: int | None = None):
        self.axis = axis
        super().__init__(x)
        self.x = x
        self.label = None

    def get_grad(self, grad):
        assert self.x.shape == self.val.shape, f"self.x shape{self.x.shape} != self.val shape {self.val.shape}"
        mask = (self.x.x == self.val)
        div = mask.sum(axis=self.axis)
        new = mask/div
        return check_shape(grad*new, self.x)

    def result(self):
        result = get_value(self.x)
        return np.amax(result, axis=self.axis) if self.axis is not None else np.max(result)


class Multi(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        labels = get_label(x, y)
        self.label = f"({labels[0]} * {labels[1]})"

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = grad*self._y_val, grad*self._x_val, f"{grad}*{self.y}", f"{grad}*{self.x}"
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return Multi(self.holder, self.y), Multi(self.holder, self.x)

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
        labels = get_label(x, y)
        self.label = f"({labels[0]} / {labels[1]})"

    def get_grad(self, grad):
        grad1 = np.array(grad)/self._y_val
        grad2 = np.array(grad)*(-self._x_val/np.square(self._y_val))
        expression1, expression2 = f"{grad}*(-1)", f"{grad}*1"
        return check_shape(grad1, self.x), check_shape(grad2, self.y), expression1, expression2

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*(1/self.y), self.holder*(-(self.x/square(self.y)))

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        self._x_val = result1
        self._y_val = result2
        return result1/result2


class exp(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"e**{labels[0]}"

    def get_grad(self, grad):
        grad = grad*np.exp(self._x_val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if not isinstance(self.holder, Matrix):
            self.holder = Matrix(self.holder)
        return self.holder*exp(self.x),

    def result(self) -> np.ndarray:
        val = get_value(self.x)
        self._x_val = val
        return np.exp(val)

    def __getitem__(self, item):
        return self.x.x[item]


class pow(Function):
    def __init__(self, x, power):
        self.power = power
        super().__init__(x, power)
        self.x = x
        labels = get_label(x)
        self.label = f"{labels[0]}**{power}"

    def get_grad(self, grad):
        grad1 = grad*self.power*np.power(self._x_val, self.power-1)
        return check_shape(grad1, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.power(val, self.power)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*self.power*pow(self.x, self.power-1),

    def gradient(self, grad=None, debug=False, create_graph=False, head=True):
        if grad is None:
            grad = 1.0
        grad = np.array(grad)
        Prime(self, grad=grad, debug=debug, create_graph=create_graph, head=True)


class sin(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"sin({labels[0]})"

    def get_grad(self, grad):
        grad1 = grad*np.cos(self._x_val)
        return check_shape(grad1, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*cos(self.x),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.sin(val)


class cos(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"cos({labels[0]})"

    def get_grad(self, grad):
        grad1 = -grad*np.sin(self._x_val)
        return check_shape(grad1, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return -self.holder*sin(self.x),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.cos(val)


class sec(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"sec({labels[0]})"

    def get_grad(self, grad):
        grad = grad*np.tan(self._x_val)*(1/np.cos(self._x_val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*tan(self.x)*sec(self.x),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return 1/np.cos(val)


class arcsin(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"arcsin({labels[0]})"

    def get_grad(self, grad):
        grad = grad*(1/np.sqrt(1-np.square(self._x_val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return -self.holder*(1/sqrt(1 - square(self.x))),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.arcsin(val)


class arcos(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"arcos({labels[0]})"

    def get_grad(self, grad):
        grad = grad*(-1/np.sqrt(self._x_val+1))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*(-1/sqrt(self.x + 1)),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.arccos(val)


class ln(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"ln({labels[0]})"

    def get_grad(self, grad):
        grad = grad*(1/self._x_val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*(1/self.x),

    def get_label(self):
        return

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
        labels = get_label(x)
        self.label = f"log({base}, {labels[0]})"

    def get_grad(self, grad):
        grad = grad*(1/(self._x_val*np.log(self.base)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*(1/(self.x*ln(self.base))),

    def result(self):
        val = get_value(self.x)
        assert np.all(val > 0), "input has element <= zero"
        self._x_val = val
        return np.log(val)/np.log(self.base)


class cot(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"cot({labels[0]})"

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*np.square(csc(val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*square(csc(self.x)),

    def result(self):
        val = get_value(self.x)
        return 1/np.tan(val)


class csc(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"csc({labels[0]})"

    def get_grad(self, grad):
        grad = grad*(-1)*np.csc(self.x.val)*cot(self.x.val).val
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*(-1)*csc(self.x)*cot(self.x),

    def result(self):
        val = get_value(self.x)
        return 1/np.sin(val)


class arcot(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"arcot({labels[0]})"

    def get_grad(self, grad):
        grad = grad*(-1/(1 + np.square(self.x.val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*(-1/(1 + square(self.x))),

    def result(self):
        val = get_value(self.x)
        return np.arctan(1/val)


class tan(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"tan({labels[0]})"

    def get_grad(self, grad):
        grad = grad*(np.square(sec(self.x.val).val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*square(sec(self.x)),

    def result(self):
        val = get_value(self.x)
        return np.tan(val)


class arctan(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"arctan({labels[0]})"

    def get_grad(self, grad):
        grad = grad*(1/(1 + np.square(self.x.val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*(1/(1 + square(self.x))),

    def result(self):
        val = get_value(self.x)
        return np.arctan(val)


class square(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"square({labels[0]})"

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad*2*val
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*2*self.x,

    def result(self):
        val = get_value(self.x)
        return np.square(val)


class sqrt(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"sqrt({labels[0]})"

    def get_grad(self, grad,):
        val = get_value(self.x)
        grad = grad*0.5*np.power(val, -0.5)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder)
        return self.holder*0.5*pow(self.x, -0.5),

    def result(self):
        val = get_value(self.x)
        return np.sqrt(val)


class argmax(Function):
    def __init__(self, x):
        super(argmax, self).__init__(x)
        self.x = x
        self.label = x.label

    def result(self):
        val = get_value(self.x)
        return np.argmax(val)

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False, create_graph=False) -> None:
        raise RuntimeError("Doesn't need gradient method")


class argmin(Function):
    def __init__(self, x):
        super(argmin, self).__init__(x)
        self.x = x

    def result(self):
        val = get_value(self.x)
        return np.argmin(val)

    def gradient(self, grad: list | float | np.ndarray | None = None, debug=False, create_graph=False) -> None:
        raise RuntimeError("Doesn't need gradient method")


class Abs(Function):
    def __init__(self, x):
        super(Abs, self).__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"abs({labels[0]})"

    def result(self):
        val = get_value(self.x)
        return np.abs(val)

    def get_grad(self, grad):
        val = (self.x.x != self.val)
        val = np.where(val == 0, 1, -1)
        return check_shape(grad*val, self.x)


class Prime:
    def __init__(self, func, grad, debug=False, create_graph=False, head=False):
        self.head = head
        self.grad = grad
        self.debug = debug
        self.create_graph = create_graph
        self.prime(func)

    def prime(self, func):

        if isinstance(func, Matrix):
            func.grad = self.grad if func.grad is None else func.grad + self.grad
            if self.debug:
                func._expression = np.array2string(func.val, prefix=f"Branch 2:  {self.__class__.__name__}(")
                print("Branch 2:", func, func.grad, id(func))

        elif isinstance(func, (Divide, Add, Multi, Sub, Matmul, starmulti)):
            grad1, grad2, expression1, expression2 = func.get_grad(self.grad)
            if self.create_graph:
                if func.graph:
                    func.holder = func.graph
                    graphs = func.get_graph()
                    func.graph = None
                else:  # first node
                    func.holder = self.grad
                    graphs = func.get_graph()
                if hasattr(func.x, 'graph'):
                    func.x.graph = graphs[0] if func.x.graph is None else func.x.graph + graphs[0]
                    func.x.graph.parent = func.x
                    func.x.son = func.x.graph
                if hasattr(func.y, 'graph'):
                    func.y.graph = graphs[1] if func.y.graph is None else func.y.graph + graphs[1]
                    func.y.graph.parent = func.y
                    func.y.son = func.y.graph
            if self.debug:
                func._expression = np.array2string(func.val, prefix=f"Branch 3:  {self.__class__.__name__}(")
                print("Branch 3:", func, grad1, grad2, id(func))
            Prime(func.x, grad1, debug=self.debug, create_graph=self.create_graph)
            Prime(func.y, grad2, debug=self.debug, create_graph=self.create_graph)

        elif type(func) in derivatives:
            grad = func.get_grad(self.grad)
            if self.debug:
                func._expression = np.array2string(func.val, prefix=f"Branch 4:  {self.__class__.__name__}(")
                print("Branch 4:", func, grad, id(func))
            if self.create_graph:
                if func.graph:
                    func.holder = func.graph
                    graphs = func.get_graph()
                    func.graph = None
                else:  # first node
                    func.holder = self.grad
                    graphs = func.get_graph()
                if hasattr(func.x, 'graph'):
                    func.x.graph = graphs[0] if func.x.graph is None else func.x.graph + graphs[0]
                    func.x.graph.parent = func.x
                    func.x.son = func.x.graph
            Prime(func.x, grad, create_graph=self.create_graph, debug=self.debug)


class View:
    def __init__(self, func, total_task, current=0, graph_object=None, head=False, parent=None, shape='box', filename='view.gv', time_begin=0.):
        self.graph_object = graph_object
        if head:
            self.graph_object = graphviz.Digraph('g', filename=filename, strict=True)
            self.graph_object.attr('node', shape=shape)
            parent = str(id(func))
        self.time = time_begin
        self.parent = parent
        self.total = total_task
        self.current_task = current
        self.head = head
        self.current_task = self.view(func)

    def view(self, func):
        if isinstance(func, (int, float, np.ndarray)):

            self.current_task += 1
            print_result(self.current_task, self.total, self.time)

            self.graph_object.node(str(id(func)), f"{func}",
                                   style='filled', fillcolor='#40e0d0')
            self.graph_object.edge(str(id(func)),
                                   self.parent)
        elif isinstance(func, Matrix):

            self.current_task += 1
            print_result(self.current_task,
                         self.total, self.time)

            self.graph_object.node(str(id(func)), f"{simplify(func.label)}\n{func.shape}",
                                   style='filled', fillcolor='#40e0d0')
            self.graph_object.edge(str(id(func)),
                                   self.parent)

        elif isinstance(func, (Divide, Add, Multi, Sub, Matmul, starmulti)):

            self.current_task += 1
            print_result(self.current_task,
                         self.total, self.time)

            string_func = str(id(func))
            label = func.label
            self.graph_object.node(name=string_func,
                                   label=func.__class__.__name__,
                                   style='filled',
                                   fillcolor=color_map[type(func)])
            if not self.head:
                self.graph_object.edge(string_func, self.parent,
                                       label=f"<{replace_upscript(str(simplify(label, doit=False)))}<BR/>{func.shape}>")
            else:
                self.graph_object.node(name=str(id(label)),
                                       label=f"<{replace_upscript(str(simplify(func.label, doit=False)))}<BR/>{func.shape}>",
                                       style='filled',
                                       fillcolor=color_map[type(func)])
                self.graph_object.edge(string_func, str(id(label)))
            v = View(func.x, graph_object=self.graph_object,
                     parent=string_func, current=self.current_task,
                     total_task=self.total, time_begin=self.time)
            v = View(func.y, graph_object=self.graph_object,
                     parent=string_func, current=v.current_task,
                     total_task=self.total, time_begin=self.time)
            self.current_task = v.current_task
        elif type(func) in derivatives:

            self.current_task += 1
            print_result(self.current_task, self.total, self.time)

            string_func = str(id(func))
            self.graph_object.node(name=string_func, label=func.__class__.__name__,
                                   style='filled', fillcolor=color_map[type(func)])
            if not self.head:
                self.graph_object.edge(string_func,
                                       self.parent,
                                       label=f"<{replace_upscript(str(simplify(func.label, doit=False)))}<BR/>{func.shape}>")
            else:
                label = str(id(func.label))
                self.graph_object.node(name=label,
                                       label=f"<{replace_upscript(str(simplify(func.label, doit=False)))}<BR/>{func.shape}>",
                                       style='filled',
                                       fillcolor=color_map[type(func)])

                self.graph_object.edge(string_func, label)

            v = View(func.x, graph_object=self.graph_object,
                     parent=string_func, current=self.current_task,
                     total_task=self.total, time_begin=self.time)

            self.current_task = v.current_task
        elif isinstance(func.x, (int, float, np.ndarray)):
            self.current_task += 1
            print_result(self.current_task, self.total, self.time)
            self.graph_object.node(str(id(func)), f"{func}")
            self.graph_object.edge(str(id(func)), self.parent)
        return self.current_task


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


def reset_graph(*var) -> list[Function, ...]:
    graphs = [i.graph for i in var]
    for i in var:
        i.grad = None
        i.graph = None
    return graphs


def change_shape(x, colms, index):
    temp = x[index]
    x[index] = temp//colms
    x.insert(len(x) + 1 + index, colms)
    return x


def get_label(*args):
    return tuple(i.label if hasattr(i, 'label') else 'x' for i in args)


def search_nodes(func, count=0):
    if isinstance(func, (int, float, np.ndarray)):
        count += 1
    elif isinstance(func, Matrix):
        count += 1
    elif isinstance(func, (Divide, Add, Multi, Sub, Matmul, starmulti)):
        count += 1
        count = search_nodes(func.x, count)
        count = search_nodes(func.y, count)
    elif type(func) in derivatives:
        count += 1
        count = search_nodes(func.x, count)
    elif isinstance(func.x, (int, float, np.ndarray)):
        count += 1
    return count


def print_result(current, total, begin):
    lis = ['[' if i == 0 else ']' if i == 21 else ' ' for i in range(22)]
    index = int((current+1)/total*20)
    percentage = format(current*100 / total, '.2f')
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


def replace_upscript(inp: str):
    string = inp.split("**")
    left = 0
    right = 0
    FLAG = False
    ls = [list(i) for i in string]
    digitmode = False
    op = ''
    for index, i in enumerate(ls):
        for idx, word in enumerate(i):
            if not FLAG and word.isdigit():
                digitmode = True
            if idx - len(i) == -1 and digitmode:
                i.append('</SUP>')
                break
            if digitmode and not word.isdigit():
                i.insert(idx, '</SUP>')
                digitmode = False
                FLAG = True
                right = 0
                left = 0
            if not digitmode and not FLAG and i == '(':
                left += 1
            if not digitmode and not FLAG and i == ')':
                right += 1
            if left > 0 and left == right:
                left = 0
                right = 0
                i.insert(idx, '</SUP>')
        FLAG = False
        if index is not len(ls) - 1:
            i.append('<SUP>')
        op += ''.join(i)
    return op


derivatives = (exp, sin, tan, sec, pow, ln, arcsin, arcos, arcot, arctan, cos, csc, cot, sqrt, square, transpose,
               trace, inv, Sum, Max, Slice, reshape, Abs, Min, mean, log, repeat)


color_map = {Divide: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Add: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Multi: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Sub: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Matmul: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             starmulti: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             exp: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             sin: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             tan: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             sec: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             pow: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             ln: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             arcsin: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             arcos: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             arcot: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             arctan: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             cos: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             csc: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             cot: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             sqrt: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             square: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             transpose: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             trace: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             inv: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Sum: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Max: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Slice: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             reshape: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Abs: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             Min: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             mean: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             log: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             repeat: f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
             }



