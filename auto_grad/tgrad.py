import threading

import cupy as np
import graphviz
import random
from time import time, time_ns
from sympy import powsimp, cancel, combsimp
import matplotlib.pyplot as plt


class Function:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.val = self.result()
        self.nth = None
        self.holder = None
        self.graph = None
        self.parent = None
        self.son = None
        self.view = None
        self.label = None
        self.shape = self.val.shape
        pass

    def update_label(self):
        pass

    @staticmethod
    def plot(*func, start=-10, end=10):
        pass

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

    def visualize(self, open_file=False, file_name=None, size=None):
        begin = time()
        total = search_nodes(self)
        p = View(self, head=True, time_begin=begin, total_task=total, filename=file_name)
        if size is not None:
            p.graph_object.graph_attr['size'] = size
        if open_file:
            p.graph_object.view()
        else:
            return p.graph_object

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 debug=False,
                 create_graph=False,
                 ):
        self.nth = 1 if self.nth is None else self.nth + 1
        if grad is None:
            grad = np.array(1.).reshape(self.val.shape)
        elif isinstance(grad, (list, float, int)):
            grad = np.array(grad)
        elif isinstance(grad, np.ndarray):
            pass
        else:
            raise TypeError("Only supports float, list and ndarray")
        Prime(self,
              nth=self.nth,
              grad=grad,
              debug=debug,
              head=True,
              create_graph=create_graph)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        elif len(shape) == 2 and isinstance(shape[0], int) and isinstance(
                shape[1], int):
            pass
        else:
            raise TypeError(
                "Invalid arguments, should be either tuple (1, 2) or 1, 2")
        return reshape(self, shape)

    def sum(self):
        pass

    def __str__(self):
        expression = np.array2string(
            self.val, prefix=f"{self.__class__.__name__}(", separator=', ')
        return f"{self.__class__.__name__}({expression}, shape={self.shape})"

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
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

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
        for index, val in enumerate(self.val):
            yield Matrix(val, f"Mat{index}")

    def __abs__(self):
        return Abs(self)

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        x = self.val[item]
        return Slice(x, self, item)

    def __min__(self):
        return Min(self)

    def __len__(self):
        return len(self.val)


class Matrix(Function):

    def __init__(self, data, label=None, label_on=False):
        x = data if isinstance(data, np.ndarray) else np.array(data)
        super().__init__(x)
        self.label = label
        if label_on and label is None:
            self.label = str(x)
        self.grad = None
        self.x = x
        self.ls = data
        self.shape = x.shape
        self.T = x.T
        self.size = x.size
        self.graph = None
        self._islast = False
        pass

    def result(self):
        return self.x

    def repeat(self, *direction):
        return repeat(self, direction)

    def __len__(self):
        return len(self.x)


class Slice(Function):

    def __init__(self, data, origin_data, index):
        super().__init__(data)
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

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 debug=False,
                 create_graph=False):
        if grad is None:
            grad = np.ones(self._shape)
        elif isinstance(grad, np.ndarray):
            pass
        else:
            grad = np.array(grad)
        Prime(self, grad=grad, debug=debug)

    def result(self):
        return get_value(self.x)


class stack(Function):
    def __init__(self, data, axis, label_on=False):
        self.axis = axis
        super().__init__(data)
        if label_on:
            labels = get_label(data)
            self.x1 = labels[0]
            self.label = f"EqualSlice({self.x1})"
        self.grad = None

    def get_grad(self, grad=None):

        pass

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        graph = (self.holder@self.y.T, self.x@self.holder)
        return graph

    def result(self):
        val = get_value(self.x)
        return np.stack(val, axis=self.axis)


class EqualSlice(Function):

    def __init__(self, origin_data, num, axis, label_on=False):
        self.num = num
        self.axis = axis
        super().__init__(origin_data)
        self._shape = origin_data.shape
        self.len = len(self.x) - 1
        self.x3 = self.get_real()
        if label_on:
            self.x1 = [i.label for i in self.x3]
            self.label = f"EqualSlice({self.x1})"
        self.total = 0
        self.grad = None
        self.cumulate = 0
        self.count2 = 0
        self.count = num

    def get_grad(self, grad=None):
        """
        :param grad: same shape as the sliced array shape
        :return: ndarray
        """
        if grad is None:
            grad = tuple(i.grad if i.grad is not None else np.zeros(i.shape) for i in self.x3)
            grad = np.stack(grad, axis=self.axis)
            grad = grad.reshape(self.x.shape)
            for i in self.x3:
                i.grad = None
            return check_shape(grad, self.x)
        else:
            grad = tuple(np.array(i) for i in grad)
            grad = np.stack(grad, axis=self.axis).reshape(self._shape)
            return check_shape(grad, self.x)

    def gradient(self,
                 grad: list | tuple | None = None,
                 debug=False,
                 create_graph=False):
        if grad is None:
            pass
        else:
            Prime(self, grad=grad, debug=debug)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        graph = (self.holder@self.y.T, self.x@self.holder)
        return graph

    def result(self):
        val = get_value(self.x)
        return np.array([np.squeeze(i) for i in np.split(val, self.num, axis=self.axis)])

    def get_real(self):
        ls = []
        for index, val in enumerate(self.val):
            a = converge(val)
            a.parent = self
            a.index = index
            a.label = f"converge{index}"
            ls.append(a)
        return tuple(ls)

    def get_count(self):
        q = 0
        for i in self.x3:
            q += i.converge
        self.count2 = q
        return q

    def __iter__(self):
        return iter(self.x3)


class reshape(Function):

    def __init__(self, data: Matrix, shape):
        self.shape = shape
        super().__init__(data)
        self.x = data
        self._shape = data.shape
        self.size = data.size

    def get_grad(self, grad):
        assert grad.size == self.size
        grad = grad.reshape(self._shape)
        return grad

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 debug=False,
                 create_graph=False):
        if grad is None:
            grad = np.ones(self._shape)
        Prime(self, grad=grad, debug=debug)
        self.x.grad = Matrix(self.x.grad)
        self.y.grad = Matrix(self.y.grad)

    def result(self):
        return np.reshape(self.x.val, self.shape)


class Matmul(Function):

    def __init__(self, x, y, label_on=False):
        super().__init__(x, y)
        self.x = x
        self.y = y
        if label_on:
            labels = get_label(x, y)
            self.label = f"Matmul({labels[0]}, {labels[1]})"

    def get_grad(self, grad):
        grad1 = grad @ self.y.T
        grad2 = self.x.T @ grad
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def result(self):
        val1, val2 = get_values(self.x, self.y)
        return val1 @ val2

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        graph = (self.holder@self.y.T, self.x@self.holder)
        return graph

    def gradient(self,
                 grad: list | None = None,
                 debug=False,
                 create_graph=False):
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

    def __init__(self, x: Matrix, y: Matrix, label_on=False):
        super().__init__(x, y)
        self.x = x
        self.y = y
        if label_on:
            labels = get_label(x, y)
            self.x1 = labels[0]
            self.x2 = labels[1]
            if isinstance(self.x, (Add, sub, Divide)) and not isinstance(self.y, (Add, sub, Divide)):
                self.label = f"({self.x1})*{self.x2}"
            elif not isinstance(self.x, (Add, sub, Divide)) and isinstance(self.y, (Add, sub, Divide)):
                self.label = f"{self.x1}*({self.x2})"
            elif isinstance(self.x, (Add, sub, Divide)) and isinstance(self.y, (Add, sub, Divide)):
                self.label = f"({self.x1})*({self.x2})"
            else:
                self.label = f"{self.x1}*{self.x2}"

    def get_grad(self, grad):
        val1, val2 = get_values(self.x, self.y)
        grad1 = val2 * grad
        grad2 = grad * val1
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        graph = (starmulti(self.y, self.holder), starmulti(self.holder, self.x))
        return graph

    def result(self):
        val1, val2 = get_values(self.x, self.y)
        return val1 * val2


class transpose(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        return check_shape(grad, self.x.shape)

    def result(self):
        val = get_value(self.x)
        return val.T

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 debug=False,
                 create_graph=False):
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
        self.shape = None

    def get_grad(self, grad):
        val = get_value(self.x)
        assert val.shape[-2] == val.shape[-1], "input has to be square matrix"
        grad = grad * np.identity(val.shape[0])
        self.shape = val.shape[0]
        return check_shape(grad, val)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder*Matrix(np.identity(self.shape), f"eye({self.shape})"),

    def result(self):
        val = get_value(self.x)
        return np.trace(val)


class inv(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"inv({self.x1})"

    def get_grad(self, grad):
        val = get_value(self.x)
        assert val.shape == grad.shape, f"shape {val.shape} != grad shape {grad.shape}"
        temp = np.linalg.inv(val)
        grad = np.transpose(temp @ np.transpose(grad) @ (-temp))  # transpose??
        return check_shape(grad, val)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return transpose(inv(self.x)@transpose(self.holder)@(Matrix(-1, "(-1)")*inv(self.x))),

    def result(self):
        val = get_value(self.x)
        op = np.linalg.inv
        self.op = op
        return op(val)


class converge(Function):
    def __init__(self, data: np.ndarray):
        super().__init__(data)
        self.grad = None
        self.x = data
        self.ls = data
        self.shape = data.shape
        self.T = np.transpose(data)
        self.size = data.size
        self.graph = None
        self._islast = False
        self.index = None
        self.label = "converge"
        self.count = 0
        self.converge = 0

    def get_grad(self, grad):
        return check_shape(grad, Matrix(self.x))

    def result(self):
        return self.x

    def repeat(self, *direction):
        return repeat(self, direction)

    def __len__(self):
        return len(self.val)

    def __iter__(self):
        for index, val in enumerate(self.val):
            yield converge(val, label=f"Converge{index}")


class Add(Function):

    def __init__(self, x, y, label_on=False):
        super().__init__(x, y)
        self.x = x
        self.y = y
        if label_on:
            labels = get_label(x, y)
            self.x1 = labels[0]
            self.x2 = labels[1]
            self.label = f"{self.x1}+{self.x2}"

    def get_grad(self, grad):
        grad1, grad2 = grad + 1e-18, grad + 1e-18
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        return np.add(result1, result2)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder, self.holder


class sum(Function):

    def __init__(self, x, axis=None, keepdims=None):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        shape = [*self.x.val.shape]
        shape[self.axis] = 1
        grad = grad.reshape(shape)
        grad = grad.repeat(self.x.val.shape[self.axis], axis=self.axis)
        grad = np.array(grad)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        direction = [1 for i in self.x.val.shape]
        direction[self.axis] = self.x.val.shape[self.axis]
        return repeat(self.holder, tuple(direction))

    def result(self):
        return np.sum(get_value(self.x), axis=self.axis, keepdims=self.keepdims)


class ScalarToMatrix(Function):
    def __init__(self, x, shape):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.shape = shape
        self.label = f"ScalarToMatrix({self.x1})"

    def get_grad(self, grad):
        return np.sum(grad)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return sum(self.holder)

    def result(self):
        result = get_value(self.x)
        return np.broadcast_to(result, self.shape)


class sub(Function):

    def __init__(self, x, y, label_on=False):
        super().__init__(x, y)
        self.x = x
        self.y = y
        if label_on:
            labels = get_label(x, y)
            self.x1 = labels[0]
            self.x2 = labels[1]
            self.label = f"{self.x1}-{self.x2}"

    def get_grad(self, grad):
        grad1, grad2, expression1, expression2 = -1 * grad, grad, f"{grad}*(-1)", f"{grad}*1"
        return check_shape(grad1, self.x), check_shape(
            grad2, self.y)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder, starmulti(self.holder, Matrix([-1.], "(-1)"))

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        self._x_val = result1
        self._y_val = result2
        return np.subtract(result1, result2)


class repeat(Function):

    def __init__(self, x, times: int, axis: int):
        self.axis = axis
        self.times = times
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        grad = np.sum(grad, axis=self.axis, keepdims=True)
        return check_shape(grad, self.x)

    def result(self):
        val = get_value(self.x)
        return np.repeat(val, self.times, axis=self.axis)


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
        new = mask / div
        return check_shape(grad * new, self.x)

    def result(self):
        result = get_value(self.x)
        return np.amax(
            result,
            axis=self.axis) if self.axis is not None else np.max(result)


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
        new = mask / div
        return check_shape(grad * new, self.x)

    def result(self):
        result = get_value(self.x)
        return np.amax(
            result,
            axis=self.axis) if self.axis is not None else np.max(result)


class Multi(Function):

    def __init__(self, x, y, label_on=False):
        super().__init__(x, y)
        self.x = x
        self.y = y
        if label_on:
            labels = get_label(x, y)
            self.x1 = labels[0]
            self.x2 = labels[1]
            if self.x1 == self.x2:
                self.label = f"{self.x1}**2"
            elif isinstance(self.x, (Add, sub, Divide)) and not isinstance(self.y, (Add, sub, Divide)):
                self.label = f"({self.x1})*{self.x2}"
            elif not isinstance(self.x, (Add, sub, Divide)) and isinstance(self.y, (Add, sub, Divide)):
                self.label = f"{self.x1}*({self.x2})"
            elif isinstance(self.x, (Add, sub, Divide)) and isinstance(self.y, (Add, sub, Divide)):
                self.label = f"({self.x1})*({self.x2})"
            elif self.x1 == "1":
                self.label = f"{self.x2}"
            elif self.x2 == "1":
                self.label = f"{self.x1}"
            else:
                self.label = f"{self.x1}*{self.x2}"

    def get_grad(self, grad):
        grad1, grad2 = grad * self._y_val, grad * self._x_val
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return Multi(self.holder, self.y), Multi(self.holder, self.x)

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        self._x_val = result1
        self._y_val = result2
        return np.multiply(result1, result2)


class Sigmoid(Function):
    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"Sigmoid({self.x1})"

    def get_grad(self, grad):
        return grad * self.val * (1 - self.val)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * self.x * (Matrix(1., "1") - self.x),

    def result(self):
        val = get_value(self.x)
        return 1 / (1 + np.exp(-val))

    @staticmethod
    def plot(*func, start=-10, end=10):
        x = np.linspace(start, end)
        y = 1 / (1 + np.exp(-x))
        plt.plot(x, y)
        plt.show()


class SoftmaxWithLoss(Function):
    def __init__(self, x, dim):
        self.dim = dim - 1
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"Softmax({self.x1})"

    def get_grad(self, grad):
        return grad * self.val * (1 - self.val)

    def result(self):
        val = get_value(self.x)
        x = val - val.max(axis=self.dim, keepdims=True)
        x_exp = np.exp(x)
        result = np.sum(x_exp, axis=self.dim, keepdims=True)
        x = x_exp / result
        return x

    @staticmethod
    def plot(*func, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.exp(x) / np.sum(np.exp(x))
        plt.plot(x.get(), y.get())
        plt.show()


class softmax(Function):
    def __init__(self, x, dim, label_on=False):
        self.dim = dim - 1
        super().__init__(x)
        self.x = x
        if label_on:
            labels = get_label(x)
            self.x1 = labels[0]
            self.label = f"Softmax({self.x1})"

    def get_grad(self, grad):
        dx = self.val * grad
        sumdx = np.sum(dx, axis=self.dim, keepdims=True)
        dx -= self.val * sumdx
        return check_shape(dx, self.x)

    def result(self):
        val = get_value(self.x)
        x = val - val.max(axis=self.dim, keepdims=True)
        x_exp = np.exp(x)
        result = np.sum(x_exp, axis=self.dim, keepdims=True)
        x = x_exp / result
        return x

    @staticmethod
    def plot(*func, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.exp(x) / np.sum(np.exp(x))
        plt.plot(x.get(), y.get())
        plt.show()


class mean(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x

    def get_grad(self, grad):
        grad = grad / np.size(self.x.x)
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
        self.x1 = labels[0]
        self.x2 = labels[1]
        if isinstance(self.x, (Add, sub)) and not isinstance(self.y, (Add, sub)):
            self.label = f"({self.x1})/{self.x2}"
        elif not isinstance(self.x, (Add, sub)) and isinstance(self.y, (Add, sub)):
            self.label = f"{self.x1}/({self.x2})"
        elif isinstance(self.x, (Add, sub)) and isinstance(self.y, (Add, sub)):
            self.label = f"({self.x1})/({self.x2})"
        else:
            self.label = f"{self.x1}/{self.x2}"

    def get_grad(self, grad):
        grad1 = np.array(grad) / (self._y_val + 1e-8)
        grad2 = np.array(grad) * (-self._x_val / (np.square(self._y_val) + 1e-8))
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder/self.y, self.holder * Matrix(-1., "(-1)")*self.x / square(self.y)

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        self._x_val = result1
        self._y_val = result2
        return result1 / result2


class inner(Function):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y
        labels = get_label(x, y)
        self.x1 = labels[0]
        self.x2 = labels[1]

    def get_grad(self, grad):
        grad1 = np.array(grad) / (self._y_val + 1e-8)
        grad2 = np.array(grad) * (-self._x_val / (np.square(self._y_val) + 1e-8))
        return check_shape(grad1, self.x), check_shape(grad2, self.y)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder/self.y, self.holder * Matrix(-1., "(-1)")*self.x / square(self.y)

    def result(self):
        result1, result2 = get_values(self.x, self.y)
        self._x_val = result1
        self._y_val = result2
        return np.inner(result1, result2)


class exp(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"exp({labels[0]})"

    def get_grad(self, grad):
        grad = grad * np.exp(self._x_val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * exp(self.x),

    def result(self) -> np.ndarray:
        val = get_value(self.x)
        self._x_val = val
        return np.exp(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.exp(x)
        plt.plot(x, y)
        plt.show()


class pow(Function):

    def __init__(self, x, power):
        self.power = power
        super().__init__(x, power)
        self.x = x
        labels = get_label(x)
        self.label = f"{labels[0]}**{power}"

    def get_grad(self, grad):
        grad1 = grad * self.power * np.power(self._x_val, self.power - 1)
        return check_shape(grad1, self.x)

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.power(val, self.power)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * self.power * pow(self.x, self.power - 1),

    @staticmethod
    def plot(self=None, power=2, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.power(x, power)
        plt.plot(x, y)
        plt.show()


class sin(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"sin({self.x1})"

    def get_grad(self, grad):
        grad1 = grad * np.cos(self._x_val)
        return check_shape(grad1, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * cos(self.x),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.sin(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.sin(x)
        plt.plot(x, y)
        plt.show()


class sec(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"sec({self.x1})"

    def get_grad(self, grad):
        grad = grad * np.tan(self._x_val) * (1 / np.cos(self._x_val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * tan(self.x) * sec(self.x),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return 1 / np.cos(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = 1 / np.cos(x)
        plt.plot(x, y)
        plt.show()


class sinh(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"sinh({self.x1})"

    def get_grad(self, grad):
        grad = grad * np.cosh(self._x_val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * cosh(self.x),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.sinh(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.sinh(x)
        plt.plot(x, y)
        plt.show()


class arcsin(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"arcsin({self.x1})"

    def get_grad(self, grad):
        grad = grad * (1 / np.sqrt(1 - np.square(self._x_val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * (Matrix(-1, "-1") / sqrt(Matrix(1, "1") - square(self.x))),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.arcsin(val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.arcsin(x)
        plt.plot(x, y)
        plt.show()


class arcsec(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"arcsec({self.x1})"

    def get_grad(self, grad):
        grad = grad * (1 / (np.abs(self._x_val)*np.sqrt(np.square(self._x_val) - 1)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder / Abs(self.x)*sqrt(square(self.x) + Matrix(-1, "(-1)")),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.arccos(1/val)

    @staticmethod
    def plot(self=None, start=-10, end=10):
        x = np.linspace(start, end)
        y = np.arccos(x)
        plt.plot(x, y)
        plt.show()


class ln(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"ln({self.x1})"

    def update_label(self):
        self.label = f"ln({self.x1})"

    def get_grad(self, grad):
        grad = grad * (1 / self._x_val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder / self.x,

    def get_label(self):
        return

    def result(self):
        val = get_value(self.x)
        assert np.all(val > 0), "input has element <= zero"
        self._x_val = val
        return np.log(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.log(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


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
        grad = grad * (1 / (self._x_val * np.log(self.base)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder/(self.x * ln(self.base)),

    def result(self):
        val = get_value(self.x)
        assert np.all(val > 0), "input has element <= zero"
        self._x_val = val
        return np.log(val) / np.log(self.base)

    @staticmethod
    def plot(base=10, *func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.log(x) / np.log(base)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class cos(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"cos({self.x1})"

    def update_label(self):
        self.label = f"cos({self.x1})"

    def get_grad(self, grad):
        grad1 = -grad * np.sin(self._x_val)
        return check_shape(grad1, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return Matrix(-1, "(-1)")*self.holder * sin(self.x),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.cos(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.cos(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class cosh(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"cosh({self.x1})"

    def get_grad(self, grad):
        grad = grad * np.sinh(self._x_val)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * sinh(self.x),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.cosh(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.cosh(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class arcos(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"arcos({self.x1})"

    def get_grad(self, grad):
        grad = grad * (-1 / np.sqrt(self._x_val + 1))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * (Matrix(-1, "(-1)") / sqrt(self.x + Matrix(1, "1"))),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.arccos(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.arccos(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class csc(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"csc({labels[0]})"

    def get_grad(self, grad):
        grad = grad * (-1) * np.csc(self._x_val) * cot(self._x_val).val
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * Matrix(-1, "(-1)") * csc(self.x) * cot(self.x),

    def result(self):
        val = get_value(self.x)
        return 1 / np.sin(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = 1 / np.sin(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class tan(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"tan({self.x1})"

    def update_label(self):
        self.label = f"tan({self.x1})"

    def get_grad(self, grad):
        grad = grad * (np.square(sec(self._x_val).val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * sec(self.x) ** 2,

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.tan(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.tan(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class arctan(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"arctan({self.x1})"

    def update_label(self):
        self.label = f"arctan({self.x1})"

    def get_grad(self, grad):
        grad = grad * (1 / (1 + np.square(self._x_val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder / (Matrix(1, "1") + square(self.x)),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.arctan(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.arctan(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class tanh(Function):

    def __init__(self, x, label_on=False):
        super().__init__(x)
        self.x = x
        if label_on:
            labels = get_label(x)
            self.x1 = labels[0]
            self.label = f"tanh({self.x1})"

    def get_grad(self, grad):
        grad = grad * (1 - np.power(self.val, 2))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * (Matrix(1, "1") - tanh(self.x)**2),

    def result(self):
        val = get_value(self.x)
        self._x_val = val
        return np.tanh(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.tanh(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class cot(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"cot({labels[0]})"

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad * np.square(csc(val))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * square(csc(self.x)),

    def result(self):
        val = get_value(self.x)
        return 1 / np.tan(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = 1 / np.tan(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class arcot(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.label = f"arcot({labels[0]})"

    def get_grad(self, grad):
        grad = grad * (-1 / (1 + np.square(self.x.val)))
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * (Matrix(-1, "(-1)") / (Matrix(1, "1") + square(self.x))),

    def result(self):
        val = get_value(self.x)
        return np.arctan(1 / val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.arctan(1 / x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class square(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"{self.x1}**2"

    def update_label(self):
        self.label = f"{self.x1}**2"

    def get_grad(self, grad):
        val = get_value(self.x)
        grad = grad * 2 * val
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * Matrix(2., "2") * self.x,

    def result(self):
        val = get_value(self.x)
        return np.square(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.square(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class sqrt(Function):

    def __init__(self, x):
        super().__init__(x)
        self.x = x
        labels = get_label(x)
        self.x1 = labels[0]
        self.label = f"sqrt({self.x1})"

    def update_label(self):
        self.label = f"sqrt({self.x1})"

    def get_grad(
        self,
        grad,
    ):
        val = get_value(self.x)
        grad = grad * 0.5 * np.power(val, -0.5)
        return check_shape(grad, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * Matrix(0.5, "0.5") * pow(self.x, -0.5),

    def result(self):
        val = get_value(self.x)
        return np.sqrt(val)

    @staticmethod
    def plot(*func, start=-10, end=10, num=300):
        graph, axis = plt.subplots()
        x = np.linspace(start, end, num)
        y = np.sqrt(x)
        axis.plot(x, y)
        for i in func:
            k = i(x)
            axis.plot(x, k.val, label=f"{k.__class__.__name__}")
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.legend()
        plt.show()


class argmax(Function):

    def __init__(self, x):
        super(argmax, self).__init__(x)
        self.x = x
        self.label = x.label

    def result(self):
        val = get_value(self.x)
        return np.argmax(val)

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 debug=False,
                 create_graph=False) -> None:
        raise RuntimeError("Doesn't need gradient method")


class argmin(Function):

    def __init__(self, x):
        super(argmin, self).__init__(x)
        self.x = x

    def result(self):
        val = get_value(self.x)
        return np.argmin(val)

    def gradient(self,
                 grad: list | float | np.ndarray | None = None,
                 debug=False,
                 create_graph=False) -> None:
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
        val = np.where(val == 0, np.array(1.), np.array(-1.))
        return check_shape(grad * val, self.x)

    def get_graph(self):
        if isinstance(self.holder, (np.ndarray, int, float)):
            self.holder = Matrix(self.holder, f"grad{self.nth}")
        return self.holder * Abs(self.x),


class Prime:

    def __init__(self,
                 func,
                 grad,
                 debug=False,
                 create_graph=False,
                 nth=1,
                 head=False,
                 stack=None):
        self.head = head
        self.grad = grad
        self.debug = debug
        self.nth = nth
        self.create_graph = create_graph
        self.stack = stack
        self.prime(func)

    def prime(self, func):

        if isinstance(func, Matrix):
            if self.create_graph:
                func.graph.nth = self.nth
            func.grad = self.grad if func.grad is None else func.grad + self.grad
            if self.debug:
                func._expression = np.array2string(
                    func.val, prefix=f"Branch 2:  {self.__class__.__name__}(")
                print("Branch 2:", func, func.grad, id(func))

        elif isinstance(func, converge):
            if self.create_graph:
                func.graph.nth = self.nth
            func.grad = func.get_grad(self.grad)
            Prime(func.parent,
                  grad=None,
                  nth=self.nth,
                  debug=self.debug,
                  create_graph=self.create_graph)
            if self.debug:
                func._expression = np.array2string(
                    func.val, prefix=f"Branch 2:  {self.__class__.__name__}(")
                print("Branch 2:", func, func.grad, id(func))

        elif isinstance(func, (Divide, Add, Multi, sub, Matmul, starmulti)):
            grad1, grad2 = func.get_grad(self.grad)
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
                    func.y.graph = graphs[
                        1] if func.y.graph is None else func.y.graph + graphs[1]
                    func.y.graph.parent = func.y
                    func.y.son = func.y.graph
            if self.debug:
                func._expression = np.array2string(
                    func.val, prefix=f"Branch 3:  {self.__class__.__name__}(")
                print("Branch 3:", func, grad1, grad2, id(func))
            Prime(func.x,
                  grad1,
                  nth=self.nth,
                  debug=self.debug,
                  create_graph=self.create_graph)
            Prime(func.y,
                  grad2,
                  nth=self.nth,
                  debug=self.debug,
                  create_graph=self.create_graph)

        elif type(func) in derivatives:
            grad = func.get_grad(self.grad)
            if self.debug:
                func._expression = np.array2string(
                    func.val, prefix=f"Branch 4:  {self.__class__.__name__}(")
                print("Branch 4:", func, grad, id(func))
            if self.create_graph:
                if func.graph:
                    func.holder = func.graph
                    graphs = func.get_graph()
                    func.graph = None
                else:  # first node
                    func.holder = self.grad
                    func.nth = self.nth
                    graphs = func.get_graph()
                if hasattr(func.x, 'graph'):
                    func.x.graph = graphs[0] if func.x.graph is None else func.x.graph + graphs[0]
                    func.x.graph.parent = func.x
                    func.x.son = func.x.graph
            Prime(func.x,
                  grad,
                  nth=self.nth,
                  create_graph=self.create_graph,
                  debug=self.debug)

        elif isinstance(func, EqualSlice):
            grad = func.get_grad(self.grad)
            if self.debug:
                func._expression = np.array2string(
                    func.val, prefix=f"Branch 4:  {self.__class__.__name__}(")
                print("Branch 4:", func, grad, id(func))
            if self.create_graph:
                if func.graph:
                    func.holder = func.graph
                    graphs = func.get_graph()
                    func.graph = None
                else:  # first node
                    func.holder = self.grad
                    func.nth = self.nth
                    graphs = func.get_graph()
                if hasattr(func.x, 'graph'):
                    func.x.graph = graphs[0] if func.x.graph is None else func.x.graph + graphs[0]
                    func.x.graph.parent = func.x
                    func.x.son = func.x.graph
            Prime(func.x,
                  grad,
                  nth=self.nth,
                  create_graph=self.create_graph,
                  debug=self.debug)


class View:

    def __init__(self,
                 func,
                 total_task,
                 current=0,
                 graph_object=None,
                 head=False,
                 parent=None,
                 shape='box',
                 filename='view',
                 time_begin=0.,
                 forward=True):
        self.graph_object = graph_object
        if head:
            self.graph_object = graphviz.Digraph('g',
                                                 filename=filename,
                                                 strict=True)
            self.graph_object.attr('node', shape=shape)
            parent = str(id(func))
        self.time = time_begin
        self.parent = parent
        self.total = total_task
        self.current_task = current
        self.head = head
        self.forward = forward
        self.current_task = self.view(func)

    def view(self, func):
        if isinstance(func, (int, float, np.ndarray)):

            self.current_task += 1
            print_result(self.current_task, self.total, self.time)
            self.graph_object.node(str(id(func)),
                                   f"{func}\n()",
                                   style='filled',
                                   fillcolor='#40e0d0')

        elif isinstance(func, converge):
            func.parent.cumulate += 1
            self.current_task += 1
            print_result(self.current_task, self.total, self.time)
            func.nth += 1
            self.graph_object.node(str(id(func)),
                                   f"{func.label}\n{func.shape}",
                                   style='filled',
                                   fillcolor='#40e0d0')
            self.graph_object.edge(str(id(func.parent)),
                                   str(id(func)),
                                   label=
                                   f"<{replace_upscript(str(func.label))}<BR/>{func.shape}>")
            View(func.parent,
                 graph_object=self.graph_object,
                 current=self.current_task,
                 total_task=self.total,
                 time_begin=self.time,
                 forward=False)

        elif isinstance(func, Matrix):

            self.current_task += 1
            print_result(self.current_task, self.total, self.time)
            func.nth += 1
            self.graph_object.node(str(id(func)),
                                   f"{func.label}\n{func.shape}",
                                   style='filled',
                                   fillcolor='#40e0d0')

        elif isinstance(func, (Divide, Add, Multi, sub, Matmul, starmulti)):
            self.current_task += 1
            print_result(self.current_task, self.total, self.time)

            string_func = str(id(func))
            label = func.label
            self.graph_object.node(name=string_func,
                                   label=func.__class__.__name__,
                                   style='filled',
                                   fillcolor=color_map[type(func)])
            v = View(func.x,
                     graph_object=self.graph_object,
                     current=self.current_task,
                     total_task=self.total,
                     time_begin=self.time)
            v = View(func.y,
                     graph_object=self.graph_object,
                     current=v.current_task,
                     total_task=self.total,
                     time_begin=self.time)
            if not self.head:
                if hasattr(func.x, 'label'):
                    new_label1 = func.x.label
                    func.x1 = new_label1
                else:
                    new_label1 = str(func.x)
                    func.x1 = new_label1
                if hasattr(func.y, 'label'):
                    new_label2 = func.y.label
                    func.x2 = new_label2
                else:
                    new_label2 = str(func.y)
                    func.x2 = new_label2
                label1 = func.x.shape if hasattr(func.x, "shape") else func.x
                label2 = func.y.shape if hasattr(func.y, "shape") else func.y
                self.graph_object.edge(
                    str(id(func.x)),
                    string_func,
                    label=
                    f"<{replace_upscript(str(powsimp(combsimp(cancel(new_label1)))))}<BR/>{label1}>")
                self.graph_object.edge(
                    str(id(func.y)),
                    string_func,
                    label=
                    f"<{replace_upscript(str(powsimp(combsimp(cancel(new_label2)))))}<BR/>{label2}>")
            else:
                optimize = str(powsimp(combsimp(cancel(func.label))))
                func._optimized_graph = compile(optimize, '<string>', 'eval')
                self.graph_object.node(
                    name=str(id(label)),
                    label=
                    f"<{replace_upscript(optimize)}<BR/>{func.shape}>",
                    style='filled',
                    fillcolor=color_map[type(func)])
                self.graph_object.edge(string_func, str(id(label)))
                if hasattr(func.x, 'label'):
                    new_label1 = func.x.label
                    func.x1 = new_label1
                else:
                    new_label1 = str(func.x)
                    func.x1 = new_label1
                if hasattr(func.y, 'label'):
                    new_label2 = func.y.label
                    func.x2 = new_label2
                else:
                    new_label2 = str(func.y)
                    func.x2 = new_label2
                label1 = func.x.shape if hasattr(func.x, "shape") else func.x
                label2 = func.y.shape if hasattr(func.y, "shape") else func.y
                self.graph_object.edge(
                    str(id(func.x)),
                    string_func,
                    label=
                    f"<{replace_upscript(str(powsimp(combsimp(cancel(new_label1)))))}<BR/>{label1}>")
                self.graph_object.edge(
                    str(id(func.y)),
                    string_func,
                    label=
                    f"<{replace_upscript(str(powsimp(combsimp(cancel(new_label2)))))}<BR/>{label2}>")
            self.current_task = v.current_task
        elif type(func) in derivatives:
            self.current_task += 1
            print_result(self.current_task, self.total, self.time)

            string_func = str(id(func))
            self.graph_object.node(name=string_func,
                                   label=func.__class__.__name__,
                                   style='filled',
                                   fillcolor=color_map[type(func)])

            v = View(func.x,
                     graph_object=self.graph_object,
                     current=self.current_task,
                     total_task=self.total,
                     time_begin=self.time)
            if not self.head:
                if hasattr(func.x, 'label'):
                    new_label1 = func.x.label
                    func.x1 = new_label1
                else:
                    new_label1 = str(func.x)
                    func.x1 = new_label1
                label1 = func.x.shape if hasattr(func.x, "shape") else func.x
                self.graph_object.edge(
                    str(id(func.x)),
                    string_func,
                    label=
                    f"<{replace_upscript(new_label1)}<BR/>{label1}>")
            else:
                optimize = str(powsimp(combsimp(cancel(func.label))))
                func._optimized_graph = compile(optimize, '<string>', 'eval')
                label = str(id(func.label))
                self.graph_object.node(
                    name=label,
                    label=
                    f"<{replace_upscript(optimize)}<BR/>{func.x.shape}>",
                    style='filled',
                    fillcolor=color_map[type(func)])
                self.graph_object.edge(string_func, label)
                new_label1 = str(func.x.label)
                func.x1 = new_label1
                label1 = func.x.shape if hasattr(func.x, "shape") else func.x
                self.graph_object.edge(
                    str(id(func.x)),
                    string_func,
                    label=
                    f"<{replace_upscript(str(powsimp(combsimp(cancel(new_label1)))))}<BR/>{label1}>")
        elif isinstance(func, EqualSlice):
            self.current_task += 1
            print_result(self.current_task, self.total, self.time)

            string_func = str(id(func))
            self.graph_object.node(name=string_func,
                                   label=func.__class__.__name__,
                                   style='filled',
                                   fillcolor=color_map[type(func)])
            v = View(func.x,
                     graph_object=self.graph_object,
                     current=self.current_task,
                     total_task=self.total,
                     time_begin=self.time)
            label1 = func.x.shape if hasattr(func.x, "shape") else func.x
            if not self.head:
                if hasattr(func.x, 'label'):
                    new_label1 = func.x.label
                    func.x1 = new_label1
                else:
                    new_label1 = str(func.x)
                    func.x1 = new_label1
                self.graph_object.edge(
                    str(id(func.x)),
                    string_func,
                    label=
                    f"<{replace_upscript(new_label1)}<BR/>{label1}>")
            else:
                optimize = str(powsimp(combsimp(cancel(func.label))))
                func._optimized_graph = compile(optimize, '<string>', 'eval')
                label = str(id(func.label))
                self.graph_object.node(
                    name=label,
                    label=
                    f"<{replace_upscript(optimize)}<BR/>{func.shape}>",
                    style='filled',
                    fillcolor=color_map[type(func)])
                self.graph_object.edge(string_func, label)
                new_label1 = str(func.x.label)
                func.x1 = new_label1
                label1 = func.x.shape if hasattr(func.x, "shape") else func.x
                self.graph_object.edge(
                    str(id(func.x)),
                    string_func,
                    label=
                    f"<{replace_upscript(str(powsimp(combsimp(cancel(new_label1)))))}<BR/>{label1}>")

            self.current_task = v.current_task
        elif isinstance(func.x, (int, float, np.ndarray)):
            self.current_task += 1
            print_result(self.current_task, self.total, self.time)
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
    if shape and np.size(grad) != np.size(inp.val) and not np.size(grad) % np.size(inp.val):
        grad = np.sum(grad, axis=0, keepdims=True)
    return grad


def reset_graph(*var) -> list[Function, ...]:
    graphs = [i.graph for i in var]
    for i in var:
        i.grad = None
        i.graph = None
    return graphs


def change_shape(x, colms, index):
    temp = x[index]
    x[index] = temp // colms
    x.insert(len(x) + 1 + index, colms)
    return x


def get_label(*args):
    return tuple(i.label if hasattr(i, 'label') else str(i) for i in args)


def search_cnodes(func, count=0):
    if isinstance(func, (Divide, Add, Multi, sub, Matmul, starmulti)):
        count = search_nodes(func.x, count)
        count = search_nodes(func.y, count)
    elif type(func) in derivatives:
        count = search_nodes(func.x, count)
    elif isinstance(func, converge):
        count += 1
    return count


def search_nodes(func, count=0):
    if isinstance(func, (int, float, np.ndarray)):
        count += 1
    elif isinstance(func, Matrix):
        count += 1
    elif isinstance(func, (Divide, Add, Multi, sub, Matmul, starmulti)):
        count += 1
        count = search_nodes(func.x, count)
        count = search_nodes(func.y, count)
    elif type(func) in derivatives:
        count += 1
        count = search_nodes(func.x, count)
    elif isinstance(func, EqualSlice):
        count += 1
        for i in func.x:
            count = search_nodes(i, count)
    elif isinstance(func.x, (int, float, np.ndarray)):
        count += 1
    return count


def print_result(current, total, begin):
    lis = ['[' if i == 0 else ']' if i == 21 else ' ' for i in range(22)]
    index = int((current + 1) / total * 20)
    percentage = format(current * 100 / total, '.2f')
    if 0 <= index < 20:
        pass
    else:
        index = 20
    if index > 0:
        for i in range(1, index + 1):
            lis[i] = u'\u25A0'
        string = ''.join(lis)
        time1 = time() - begin
        print(f'\r{string} {percentage}% Time: {time1:.3f}s',
              end='',
              flush=True)
    else:
        string = ''.join(lis)
        time1 = time() - begin
        print(f'\r{string} {percentage}% Time: {time1:.3f}s',
              end='',
              flush=True)


def replace_upscript(inp: str):
    string = inp.split("**")
    left = 0
    right = 0
    FLAG = False
    ls = [list(i) for i in string]
    digitmode = False
    op = ''
    for index, i in enumerate(ls):
        i.append('<SUP>')
    ls[-1].pop()
    for index, i in enumerate(ls):
        if index > 0:
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
        op += ''.join(i)
    return op


derivatives = (exp, sin, tan, sec, pow, ln, arcsin, arcos, arcot, arctan, cos,
               csc, cot, sqrt, square, transpose, trace, inv, sum, Max, Slice,
               reshape, Abs, Min, mean, log, repeat, tanh, sinh, cosh, Sigmoid,
               ScalarToMatrix, softmax)

color_map = {
    Divide:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Add:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Multi:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sub:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Matmul:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    starmulti:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    exp:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sin:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    tan:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sec:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    pow:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    ln:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    arcsin:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    arcos:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    arcot:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    arctan:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    cos:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    csc:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    cot:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sqrt:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    square:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    transpose:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    trace:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    inv:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    sum:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Max:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Slice:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    reshape:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Abs:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Min:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    mean:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    log:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    repeat:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    tanh:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    EqualSlice:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    Sigmoid:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
    ScalarToMatrix:
    f"{random.uniform(0.65, 1)} {random.uniform(0.65, 1)} {random.uniform(0.65, 1)}",
}
