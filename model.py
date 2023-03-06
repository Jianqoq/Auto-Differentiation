from layers2 import MatMul, Softmax_Cross_entropy, WordEmbed, WordEmbedDot, SigmoidWithLoss
import cupy as np


class SimpleCbow:
    def __init__(self, row, cols):
        W_in = np.random.randn(row, cols).astype('f')
        W_out = np.random.randn(cols, row).astype('f')
        self.layer1 = MatMul(W_in)
        self.layer2 = MatMul(W_in)
        self.layer3 = MatMul(W_out)
        self.layer4 = Softmax_Cross_entropy()

        self.params, self.grads = [], []
        self.params += self.layer1.weights
        self.params += self.layer2.weights
        self.params += self.layer3.weights
        self.grads += self.layer1.gradients
        self.grads += self.layer2.gradients
        self.grads += self.layer3.gradients

    def forward(self, context, target):

        a = self.layer1.forward(context[:, 0])
        b = self.layer2.forward(context[:, 1])
        c = 0.5 * (a + b)
        prediction = self.layer3.forward(c)
        loss = self.layer4.forward(target, prediction)

        return loss

    def backward(self, dout=1):

        ds = self.layer4.backward(dout)
        da = self.layer3.backward(ds)
        da *= 0.5
        self.layer2.backward(da)
        self.layer1.backward(da)


class CBow:
    def __init__(self, row: int, cols: int, sample_size: int, sample_set):
        W_in = np.random.randn(row, cols)
        W_out = np.random.randn(row, cols)
        self.sample_set = sample_set
        self.layer1 = WordEmbed(W_in)
        self.layer2 = WordEmbed(W_in)
        self.layer3 = WordEmbed(W_out)
        self.Embedding_dot = [WordEmbedDot(W_out) for _ in range(sample_size+1)]
        self.Sigmoid_loss = [SigmoidWithLoss(W_out) for _ in range(sample_size+1)]
        self.weights, self.grads = [], []

        for i in self.Embedding_dot:
            self.weights += i.weights
            self.grads += i.grads

    def forward(self, index1, index2):
        result1 = self.layer1.forward(index1)
        result2 = self.layer2.forward(index2)
        h = 0.5*(result2+result1)
        ls = [i.forward(h, self.sample_set[index]) for index, i in enumerate(self.Embedding_dot)]
        ls2 = [i.forward(i.params[0][self.sample_set[index]], ls[index]) for index, i in enumerate(self.Sigmoid_loss)]
        loss = sum(ls2)
        return loss

    def backward(self, dout):
        pass

