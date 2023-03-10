import cupy as cp
from utils.activation_func import multi_cross_entropy_error


class MatMul:

    def __init__(self, W):
        self.weights = [W]
        self.X = None
        self.gradients = [cp.zeros_like(W)]

    def forward(self, forward_input):
        W, = self.weights
        output = cp.dot(forward_input, W)
        self.X = forward_input
        return output

    def backward(self, d_backward_input):
        # get weights and calculate dX
        W = self.weights[0]
        dX = cp.dot(d_backward_input, W.T)

        # use stored input to and dinput to calculate dW and store to self.gradients list
        dW = cp.dot(self.X.T, d_backward_input)
        self.gradients[0][...] = dW

        return dX


class Affine:
    def __init__(self, w, b):
        self.params = [w, b]
        self.grads = [cp.zeros_like(w), cp.zeros_like(b)]
        self.X = None

    def forward(self, x):
        W, b = self.params
        self.X = x
        return cp.dot(x, W) + b

    def backward(self, dout):
        w, b = self.params
        dx = cp.dot(dout, w.T)
        dw = cp.dot(self.X.T, dout)
        db = cp.sum(dout, axis=0)

        self.grads[0][...] = dw
        self.grads[1][...] = db

        return dx


class Repeat:
    def __init__(self, offset_var, repeat_num, axis: str):
        axis = axis.lower()
        if axis == 'y':
            self.repeated = cp.repeat(offset_var, repeat_num, axis=0)
        elif axis == 'x':
            self.repeated = cp.repeat(offset_var, repeat_num, axis=1)
        else:
            raise ValueError('axis should be x or y')


class Sigmoid:
    def __init__(self):
        self.X = None

    def forward(self, x) -> cp.ndarray:
        self.X = 1 / (1 + cp.exp(-x))
        return self.X

    def backward(self, dout) -> cp.ndarray:
        return dout * self.X * (1 - self.X)


class Sum:
    def forward(self, x) -> cp.ndarray:
        return cp.sum(x, axis=0, keepdims=True)

    def backward(self, dy, collums) -> cp.ndarray:
        return cp.repeat(dy, collums, axis=0)


class Softmax:

    def forward(self, x) -> cp.ndarray:
        sm = cp.sum(cp.exp(x))
        return cp.exp(x) / sm

    def backward(self, dy) -> cp.ndarray:
        pass


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = cp.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - cp.max(x)
        x = cp.exp(x) / cp.sum(cp.exp(x))

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
        loss = multi_cross_entropy_error(self.y, self.x)
        return loss

    def backward(self, dout=1):
        batch_size = self.x.shape[0]
        dx = self.y.copy()
        dx[cp.arange(batch_size), self.x] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


class WordEmbed:
    def __init__(self, W):
        self.weights = [W]
        self.grads = [cp.zeros_like(W)]
        self.index = None

    def forward(self, index) -> cp.ndarray:
        self.index = index
        w, = self.weights
        return w[index]

    def backward(self, dh) -> None:
        dw, = self.grads
        dw[...] = 0
        for i, word_id in enumerate(self.index):
            dw[word_id] += dh[i]


class WordEmbedDot:
    def __init__(self, W):
        self.embed = WordEmbed(W)
        self.weights = self.embed.weights
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, index) -> cp.ndarray:
        target = self.embed.forward(index)
        self.cache = (h, target)
        return cp.sum(target * h, axis=1)

    def backward(self, dout) -> cp.ndarray:
        h, target = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        self.embed.backward(dout*h)
        dh = dout*target
        return dh


class SigmoidWithLoss:
    def __init__(self):
        self.x, self.y = None, None

    def forward(self, true, predict) -> cp.ndarray:
        """
        accept predict and then use Sigmoid to convert it to possibility

        P = possibility, 1e-7 epsilon, label = 0 or 1\n

        For binary classcification\n

        loss=-(label*cp.log(P+1e-7)+(1-label)*cp.log(1-P+1e-7))
        """
        y = Sigmoid().forward(predict)
        self.y = y
        self.x = true
        loss = -(true*cp.log(y + 1e-7)+(1-true)*cp.log(1-y+1e-7))
        return loss

    def backward(self) -> cp.ndarray:
        return (self.y - self.x)/self.x.shape[0]


class RNN:
    def __init__(self, w_input, w_prev, b):
        self.weights, self.grads = [w_input, w_prev, b], []
        self.Matmul1 = MatMul(w_input)
        self.Matmul2 = MatMul(w_prev)
        self.cache = None

    def forward(self, x, h_prev):
        w_input, w_prev, b = self.weights
        new_h = self.Matmul1.forward(h_prev)
        new_x = self.Matmul2.forward(x)
        first_total = new_x + new_h
        # repeat = Repeat()
        final = first_total + b
        h_next = cp.tanh(final)
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        x, h_prev, h_next = self.cache
        dt = dh_next*(1 - cp.square(h_next))
        db = cp.sum(dt, axis=0)
        dx = self.Matmul1.backward(db)
        dh_next = self.Matmul2.backward(db)
        return dh_next, dx

