from layers import MatMul, Softmax_Cross_entropy, WordEmbed, WordEmbedDot, SigmoidWithLoss
import cupy as cp


class SimpleCBow:
    def __init__(self, row, cols):
        W_in = cp.random.randn(row, cols).astype('f')
        W_out = cp.random.randn(cols, row).astype('f')
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
        W_in = 0.01*cp.random.randn(row, cols).astype('f')
        W_out = 0.01*cp.random.randn(row, cols).astype('f')
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

    def forward(self, context, target, batch) -> cp.ndarray:
        result1 = self.layer1.forward(context[:, 0])
        result2 = self.layer2.forward(context[:, 1])
        h = 0.5*(result2+result1)
        batch_size = len(target)
        score = self.Embedding_dot[0].forward(h, target)
        correct_label = cp.ones(batch_size, dtype="int32")
        negative_label = cp.zeros(batch_size, dtype="int32")
        loss = self.Sigmoid_loss[0].forward(correct_label, score)
        negative_score = [self.Embedding_dot[index+1].forward(h, batch[:, index]) for index in range(len(self.Embedding_dot) - 1)]
        for index in range(len(self.Sigmoid_loss)-1):
            loss += self.Sigmoid_loss[index + 1].forward(negative_label, negative_score[index])
        return sum(loss)

    def backward(self) -> None:
        dh = 0
        for Sigmoid_loss, Embedding_dot in zip(self.Sigmoid_loss, self.Embedding_dot):
            dscore = Sigmoid_loss.backward()
            dh += Embedding_dot.backward(dscore)
        dh /= 2
        self.layer1.backward(dh)
        self.layer2.backward(dh)
