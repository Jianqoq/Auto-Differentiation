import cupy as np
from functions import cross_entropy_error


class MatMul:

    def __init__(self, W):
        self.weights = [W]
        self.X = None
        self.gradients = [np.zeros_like(W)]

    def forward(self, forward_input):
        W, = self.weights
        output = np.dot(forward_input, W)
        self.X = forward_input

        return output

    def backward(self, d_backward_input):
        # get weights and calculate dX
        W = self.weights[0]
        dX = np.dot(d_backward_input, W.T)

        # use stored input to and dinput to calculate dW and store to self.gradients list
        dW = np.dot(self.X.T, d_backward_input)
        self.gradients[0][...] = dW

        return dX


class Affine:
    def __init__(self, w, b):
        self.params = [w, b]
        self.grads = [np.zeros_like(w), np.zeros_like(b)]
        self.X = None

    def forward(self, x):
        W, b = self.params
        self.X = x
        return np.dot(x, W) + b

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dw = np.dot(self.X.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dw
        self.grads[1][...] = db

        return dx


class Repeat:
    def __init__(self, offset_var, repeat_num, axis: str):
        axis = axis.lower()
        if axis == 'y':
            self.repeated = np.repeat(offset_var, repeat_num, axis=0)
        elif axis == 'x':
            self.repeated = np.repeat(offset_var, repeat_num, axis=1)
        else:
            raise ValueError('axis should be x or y')


class Sigmoid:
    def __init__(self):
        self.X = None

    def forward(self, x):
        self.X = 1 / (1 + np.exp(-x))
        return self.X

    def backward(self, dout):
        return dout * self.X * (1 - self.X)


class Sum:
    def forward(self, x):
        return np.sum(x, axis=0, keepdims=True)

    def backward(self, dy, colloms):
        return np.repeat(dy, colloms, axis=0)


class Softmax:

    def forward(self, x):
        sm = np.sum(np.exp(x))
        return np.exp(x) / sm

    def backward(self, dy):
        pass

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


class Softmax_Cross_entropy:
    def __init__(self):
        self.params, self.grads = [], []
        self.x, self.y = None, None

    def forward(self, true, predict):
        self.x = true
        self.y = softmax(predict)
        if self.x.size == self.y.size:
            self.x = self.x.argmax(axis=1)
        loss = cross_entropy_error(self.y, self.x)
        return loss

    def backward(self, dout=1):
        batch_size = self.x.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.x] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


class WordEmbed:
    def __init__(self, W):
        self.weights = [W]
        self.grads = [np.zeros_like(W)]
        self.index = None

    def forward(self, index):
        self.index = index
        W, = self.weights
        return W[index]

    def backward(self, dh):
        dW, = self.grads
        dW[...] = 0
        for i, word_id in enumerate(self.index):
            dW[word_id] += dh[i]


class WordEmbedDot:
    def __init__(self, W):
        self.embed = WordEmbed(W)
        self.weights = self.embed.weights
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, index):
        target = self.embed.forward(index)
        self.cache = (h, target)
        return np.sum(target*h, axis=1)

    def backward(self, dout):
        h, target = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout*h
        self.embed.backward(dtarget_W)
        dh = dout*target
        return dh


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.x, self.y = None, None

    def forward(self, true, predict):
        y = Sigmoid().forward(predict)
        self.y = y
        self.x = true
        loss = cross_entropy_error(self.y, self.x)
        return loss

    def backward(self):
        return self.y - self.x