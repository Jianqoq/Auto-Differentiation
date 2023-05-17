
import numpy as np
import ad

import torch

np.random.seed(42)

class MultiHeadAttention:
    def __init__(self, hidden_size, num_heads, w1, w2, w3, w4):
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        assert isinstance(self.d_k, int), f"{self.d_k} is not int"
        self.weights = [w1, w2, w3, w4]
        self.cache = None

    def forward(self, query, key, value):
        batch_size, sequence_length, wordvec_size = query.shape
        num_heads = self.num_heads
        d_k = self.d_k
        wq, wk, wv, wo = self.weights

        _query = query @ wq
        _key = key @ wk
        _value = value @ wv
        _queryHead = _query.reshape((batch_size, sequence_length, num_heads, d_k)).permute(0, 2, 1, 3)
        _keyHead = _key.reshape(batch_size, sequence_length, num_heads, d_k).permute(0, 2, 1, 3)
        _valueHead = _value.reshape(batch_size, sequence_length, num_heads, d_k).permute(0, 2, 1, 3)
        #
        attention_scores = (_queryHead @ _keyHead.permute(0, 1, 3, 2)) / torch.sqrt(torch.Tensor([d_k]))
        attention_weights = torch.exp(attention_scores) / torch.sum(torch.exp(attention_scores), dim=-1, keepdim=True)
        attention_output = attention_weights @ _valueHead

        attention_output_concat = attention_output.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        output = attention_output_concat @ wo

        self.cache = (wq, wk, wv, wo, output, query, key, value)

        return output

    def backward(self, dout):
        wq, wk, wv, wo, output, query, key, value = self.cache

        output.backward(dout)

        return wq.grad, wk.grad, wv.grad


class AutoGradientMultiHeadAttention:
    def __init__(self, hidden_size, num_heads, w1, w2, w3, w4):
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads
        assert isinstance(self.d_k, int), f"{self.d_k} is not int"
        self.weights = [w1, w2, w3, w4]
        self.cache = None

    def forward(self, query, key, value):
        batch_size, sequence_length, wordvec_size = query.shape
        num_heads = self.num_heads
        d_k = self.d_k
        wq, wk, wv, wo = self.weights

        query = ad.Matrix(query)
        key = ad.Matrix(key)
        value = ad.Matrix(value)
        wq = ad.Matrix(wq)
        wk = ad.Matrix(wk)
        wv = ad.Matrix(wv)
        wo = ad.Matrix(wo)
        _query = ad.tensordot(query, wq, axes=([-1], [-2]))
        _key = ad.tensordot(key, wk, axes=([-1], [-2]))
        _value = ad.tensordot(value, wv, axes=([-1], [-2]))
        _queryHead = _query.reshape(batch_size, sequence_length, num_heads, d_k).transpose(0, 2, 1, 3)
        _keyHead = _key.reshape(batch_size, sequence_length, num_heads, d_k).transpose(0, 2, 1, 3)
        _valueHead = _value.reshape(batch_size, sequence_length, num_heads, d_k).transpose(0, 2, 1, 3)
        attention_scores = (_queryHead @ _keyHead.transpose(0, 1, 3, 2)) / ad.sqrt(d_k)
        exp = ad.exp(attention_scores)
        attention_weights = ad.exp(attention_scores) / ad.sum(exp, axis=-1, keepdims=True)
        attention_output = attention_weights @ _valueHead
        attention_output_concat = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        output = ad.tensordot(attention_output_concat, wo, axes=([2], [0]))

        self.cache = (wq, wk, wv, wo, output, query, key, value, output)
        return output

    def backward(self, dout):
        wq, wk, wv, wo, output, query, key, value, output = self.cache

        output.gradient(dout)

        return wq.grad, wk.grad, wv.grad

np.random.seed(42)
batch_size = 10
sentence_length = 30
wordvec_size, hidden_size, num_heads = 16, 20, 4
#
w1 = np.random.randn(wordvec_size, hidden_size)
w2 = np.random.randn(wordvec_size, hidden_size)
w3 = np.random.randn(wordvec_size, hidden_size)
w4 = np.random.randn(hidden_size, hidden_size)
autograd_version = AutoGradientMultiHeadAttention(wordvec_size, hidden_size, num_heads, w1, w2, w3, w4)
w1 = torch.Tensor(w1).double()
w2 = torch.Tensor(w2).double()
w3 = torch.Tensor(w3).double()
w4 = torch.Tensor(w4).double()
w1.requires_grad = True
w2.requires_grad = True
w3.requires_grad = True
w4.requires_grad = True
real_version = MultiHeadAttention(wordvec_size, hidden_size, num_heads, w1, w2, w3, w4)
query = np.random.randn(batch_size, sentence_length, wordvec_size)
key = np.random.randn(batch_size, sentence_length, wordvec_size)
value = np.random.randn(batch_size, sentence_length, wordvec_size)
val1 = autograd_version.forward(query=query, key=key, value=value)
val2 = real_version.forward(query=torch.Tensor(query).double(), key=torch.Tensor(key).double(), value=torch.Tensor(value).double())
autograd1, autograd2, autograd3 = autograd_version.backward(val1.val)
grad1, grad2, grad3 = real_version.backward(torch.Tensor(val1.val).double())
