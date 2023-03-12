import cupy as cp
from utils.activation_func import multi_cross_entropy_error
import numpy as np

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
    """
    for one hot encoding
    """
    def __init__(self):
        self.x, self.y = None, None

    def forward(self, true, predict):
        self.x = true
        self.y = softmax(predict)
        # if self.x.size == self.y.size:
        #     self.x = self.x.argmax(axis=1)
        loss = multi_cross_entropy_error(self.y, self.x)
        return loss

    def backward(self, dout=1):
        batch_size = self.x.shape[0]
        dx = self.y.copy()
        dx[cp.arange(batch_size), self.x] -= 1
        dx = dx / batch_size
        return dx


class SoftmaxWithLoss:
    """
    general SoftmaxWithLoss
    """
    def __init__(self):
        self.x, self.y = None, None

    def forward(self, x, true):
        score = softmax(x)
        loss = multi_cross_entropy_error(score, true)
        self.y = score
        self.x = true
        return loss

    def backward(self):
        return self.y - self.x


class WordEmbed:
    def __init__(self, W):
        self.weights = [W]
        self.grads = [cp.zeros_like(W)]
        self.index = None

    def forward(self, index) -> cp.ndarray:
        self.index = index
        w, = self.weights
        return w[index]

    def backward(self, dh: cp.ndarray) -> None:
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
    """
    :param w_input: shape(vocab_size, hidden_size)
    :param w_prev: shape(hidden_size, hidden_size)
    """
    def __init__(self, w_input, w_prev, b):
        self.weights, self.grads = [w_input, w_prev, b], []
        self.Matmul1 = MatMul(w_prev)
        self.Matmul2 = MatMul(w_input)
        self.cache = None

    def forward(self, word_vector, h_prev):
        """
        :param word_vector: shape(mini_batch_size, word_vector)
        :param h_prev: shape(mini_batch_size, hidden_size)

        new_h = shape(mini_batch_size, hidden_size)
        new_x = shape(mini_batch_size,

        :return:
        """

        w_input, w_prev, b = self.weights
        new_h = self.Matmul1.forward(h_prev)
        new_x = self.Matmul2.forward(word_vector)
        first_total = new_x + new_h
        # repeat = Repeat()
        final = first_total + b
        h_next = np.tanh(final)
        self.cache = (word_vector, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        x, h_prev, h_next = self.cache
        dt = dh_next*(1 - cp.square(h_next))
        db = cp.sum(dt, axis=0)
        dx = self.Matmul1.backward(db)
        dh_next = self.Matmul2.backward(db)
        return dh_next, dx


class TimeRNN:
    def __init__(self, w_input, w_prev, b, stateful=False):
        self.weights = [w_input, w_prev, b]
        self.grads = [cp.zeros_like(w_input), cp.zeros_like(w_prev), cp.zeros_like(b)]
        self.stateful = stateful
        self.layers = None
        self.h, self.dh = None, None  # self.h = RNN output

    def forward(self, x_sequence):
        """
        n: batch size
        """
        w_cols, hidden_size = self.weights[0].shape
        mini_batch, number_words, w_cols = x_sequence.shape
        hs = cp.empty((mini_batch, number_words, hidden_size), dtype='f')
        self.layers = []
        if not self.stateful or self.h is None:
            self.h = cp.zeros((mini_batch, hidden_size), dtype='f')  # not store previous RNN output
        for i in range(number_words):
            layer = RNN(*self.weights)
            self.h = layer.forward(x_sequence[:, i, :], self.h)  # x_sequence[:, i, :] = word vector
            self.layers.append(layer)
            hs[:, i, :] = self.h
        return hs

    def backward(self, dhs):
        wx, wh, b = self.weights
        n, t, d = dhs.shape
        dxs = cp.empty((n, t, wx.shape[0]), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for i in reversed(range(t)):
            layer = self.layers[i]
            dh, dx = layer.backward(dhs[:, i, :] + dh)
            dxs[:, i, :] = dx
            for index, k in enumerate(layer.grads):
                grads[index] += k
        for i, grad in grads:
            self.grads[i][...] = grad
        self.dh = dh
        return dxs


class TimeSigmoidWithLoss:
    def __init__(self, length):
        self.layers = []
        for i in range(length):
            self.layers.append(SoftmaxWithLoss())

    def forward(self, x_sequence, t_sequence):
        total_loss = 0
        for index, i in enumerate(zip(x_sequence, t_sequence)):
            total_loss += self.layers[index].forward(*i)
        return total_loss/x_sequence.shape[1]

    def backward(self):
        return cp.array([i.backward() for i in self.layers])


class TimeAffine:
    def __init__(self, w, b):
        self.weights = [w, b]
        self.x = None

    def forward(self, x_sequence):
        """
        :param x_sequence: same as the input of x_sequence in TimeRNN Layer

        x = np.array([[[1,2,3],[3,4,5]],[[5,6,7],[7,8,9]]])
        x = [[[1 2 3]   x.shape = (2, 2, 3)
            [3 4 5]]
            [[5 6 7]
            [7 8 9]]]

        y = np.array([[1, 2, 3, 1, 1, 1],[3, 4, 5, 1, 1, 1],[5, 6, 7, 1, 1, 1]])
        y = [[1 2 3 1 1 1]  y.shape = (3, 6)
            [3 4 5 1 1 1]
            [5 6 7 1 1 1]]

        np.dot(x, y) = [[[ 22,  28,  34,   6,   6,   6],
                       [ 40,  52,  64,  12,  12,  12]],
                       [[ 58,  76,  94,  18,  18,  18],
                       [ 76, 100, 124,  24,  24,  24]]]
        y[:,0] = [1, 3, 5]
        x[0,0,:] = [1, 2, 3]
        sum(x[0,0,:]*y[:,0]) = 22
        sum(x[0,0,:]*y[:,1]) = 28

        """
        # efficient way
        w, b = self.weights
        self.x = x_sequence
        out = cp.dot(x_sequence, w) + b
        return out
        # easy way and understandable way
        # vocab_size = self.weights[0].shape[1]
        # mini_batch, number_words, hidden_size = x_sequence.shape
        # out = np.empty((mini_batch, number_words, vocab_size), dtype='f')
        # for index, i in enumerate(self.layers):
        #     out[:, index, :] = i.forward(x_sequence[:, index, :])
        # return out

    def backward(self, d_sequence):
        for i in self.layers:
            i.backward(d_sequence)


class TimeEmbedding:
    def __init__(self, w, length):
        self.layers = []
        self.weights = [w]
        self.w = w
        self.layers = [WordEmbed(w) for _ in range(length)]

    def forward(self, x_sequence) -> cp.ndarray:
        """
        word_vector = shape(1, w.shape[1])
        :param x_sequence: cp.array([[1, 2, 3], [4, 5, 6]]) shape(2, 3) | 2 = mini_batch, 3 = number of words
        :return: cp.ndarray
        """
        w = self.w
        mini_batch, number_words = x_sequence.shape
        w_rows, w_cols = w.shape
        out = cp.empty((mini_batch, number_words, w_cols), dtype='f')
        for index, i in enumerate(self.layers):
            out[:, index, :] = i.forward(x_sequence[:, index])
            # x_sequence[:, 1]=[[1], [4]] | shape(2, ) | 2=mini_batch
            # i.forward(x_sequence[:, index]) = cp.array([[layer.weights[1]], [layer.weights[4]]]) | shape(2, ?)
        return out

    def backward(self, d_sequence):
        for i in self.layers:
            i.backward(d_sequence)


class TimeSoftmaxWithLoss:
    def __init__(self, length):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1
        self.layers = [SoftmaxWithLoss() for _ in range(length)]
        self.length = length
    def forward(self, x_sequence, label_sequence):
        """
        vocab_size = corpus_size

        :param x_sequence: predict set
        :param label_sequence: label set
        :return: ndarray
        """
        loss = 0
        for index, layer in enumerate(self.layers):
            loss += layer.forward(x_sequence[:, index, :], label_sequence[:, index])
        return loss/self.length

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[cp.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, cp.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx