from layers2 import MatMul, Softmax_Cross_entropy, WordEmbed, WordEmbedDot, SigmoidWithLoss
import numpy as np
import time




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
        self.negative_sample_set = sample_set
        self.layer1 = WordEmbed(W_in)
        self.layer2 = WordEmbed(W_in)
        self.layer3 = WordEmbed(W_out)
        self.Embedding_dot = [WordEmbedDot(W_out) for _ in range(sample_size+1)]
        self.Sigmoid_loss = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.weights, self.grads = [], []

        self.in_layers = [self.layer1, self.layer2, self.layer3]
        self.in_layers += self.Embedding_dot
        for i in self.in_layers:
            self.weights += i.weights
            self.grads += i.grads

    def forward(self, context, target: list):
        result1 = self.layer1.forward(context[:, 0])
        result2 = self.layer2.forward(context[:, 1])
        h = 0.5*(result2+result1)
        batch_size = len(target)
        score = self.Embedding_dot[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype="uint8")
        negative_label = np.zeros(batch_size, dtype="uint8")
        loss = self.Sigmoid_loss[0].forward(correct_label, score)
        negative_score = [self.Embedding_dot[index+1].forward(h, self.negative_sample_set[:, index]) for index in range(len(self.Embedding_dot)-1)]
        for index in range(len(self.Sigmoid_loss)-1):
            loss += self.Sigmoid_loss[index + 1].forward(negative_label, negative_score[index])
        return loss

    def backward(self):
        dh = 0
        for Sigmoid_loss, Embedding_dot in zip(self.Sigmoid_loss, self.Embedding_dot):
            dscore = Sigmoid_loss.backward()
            dh += Embedding_dot.backward(dscore)
        return dh
