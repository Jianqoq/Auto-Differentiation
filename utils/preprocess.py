from time import time
import numpy as np
from utils.useful_func import print_result
from typing import Optional


class Preprocess:
    def __init__(self, text: str, *args):
        dictionary = {i: f' {i}' for i in args}
        text = text.lower()
        for i in dictionary:
            text = text.replace(i, dictionary.get(i))
        self.text = text.split(' ')
        self.repeated = []

    def get_word_id(self):
        dictionary = {}
        dictionary2 = {}
        corpus = []
        append = corpus.append
        counter = 0
        for index, i in enumerate(self.text):
            if i not in dictionary:
                dictionary[i] = counter
                dictionary2[counter] = i
                counter += 1
                append(dictionary[i])
            else:
                append(dictionary[i])
                self.repeated.append(index)
        return dictionary, dictionary2, corpus

    def get_single_context(self, id_word: dict, word_id: dict, corpus: list,
                           word: str, window: int):  # list bound check
        text = self.text
        word = word.lower()
        length = len(text)
        if word not in text:
            return
        ls = [0] * len(corpus)
        for index, i in enumerate(text):
            if word_id[i] == word_id[word]:
                if index == 0:
                    counter = 1
                    for k in range(window):
                        ls[counter] += 1
                        counter += 1
                elif index == length - 1:
                    counter = 1
                    for p in range(window):
                        ls[-1 - counter] += 1
                        counter += 1
                else:
                    counter = counter2 = 1
                    word1_id = word_id[text[index - counter]]
                    word2_id = word_id[text[index + counter2]]
                    for p in range(window):
                        ls[word1_id] += 1
                        ls[word2_id] += 1
                        counter += 1
                        counter2 += 1

        return np.array(ls, dtype='uint8')

    def get_coocurrenceMatrix(self, corpus: list, id_word: dict, word_id: dict,
                              window: int):
        ls = []
        append = ls.append
        total = len(word_id)
        begin = time()
        for index, i in enumerate(word_id):
            append(self.get_single_context(id_word, word_id, corpus, i,
                                           window))
            print_result(index + 1, total, begin)
        return np.array(ls, dtype='uint8'), ls

    def PPMI(self, co_matrix, verbose=True):
        ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float32)
        N = np.sum(co_matrix)
        sigle_word = np.sum(co_matrix, axis=0)
        total = co_matrix.shape[0] * co_matrix.shape[1]
        cnt = 0
        begin = time()
        for i in range(co_matrix.shape[0]):
            for j in range(co_matrix.shape[1]):
                ppmi = np.log2(co_matrix[i, j] * N / (sigle_word[i] * sigle_word[j]) + 1e-8)
                ppmi_matrix[i, j] = max(0, ppmi)
                if verbose:
                    cnt += 1
                    if cnt % (total // 200) == 0:
                        print_result(cnt + 1, total, begin)
        return ppmi_matrix

    def create_context_target(self, corpus, windowsize=1) -> tuple[np.ndarray, np.ndarray]:
        target = corpus[1:-1]
        context = []
        cs = []
        for i in range(windowsize, len(corpus) - 1):
            cs.append(corpus[i - 1])
            cs.append(corpus[i + 1])
            context.append(cs)
            cs = []
        return np.array(context, dtype='int32'), np.array(target, dtype='int32')

    def convert_onehot(self, context, target, length) -> tuple[np.ndarray, np.ndarray]:
        zero_context = np.zeros(shape=(*context.shape, length), dtype='uint8')
        zero_target = np.zeros(shape=(*target.shape, length), dtype='uint8')
        for index, i in enumerate(context):
            for index2, k in enumerate(i):
                zero_context[index, index2, k] = 1
        for index, i in enumerate(target):
            zero_target[index, i] = 1
        return zero_context, zero_target

    def most_similar(self, matrix: list, word: str,
                     word_id: dict, id_word: dict,
                     top: int) -> Optional[list]:
        word = word.lower()
        if word not in word_id:
            return
        word_use_vector = matrix[word_id[word]]
        ls = {
            id_word[index]: self.similarity(word_use_vector, i)
            for index, i in enumerate(matrix) if index is not word_id[word]
        }
        return sorted(ls.items(), key=lambda x: x[1], reverse=True)[:top]

    def similarity(self, vect1, vect2) -> np.ndarray:
        x = vect1 / (np.sqrt(np.sum(vect1 ** 2)) + 1e-8)
        y = vect2 / (np.sqrt(np.sum(vect2 ** 2)) + 1e-8)
        return np.dot(x, y)

    def get_negative_sample(self, sample_size: int, word_id: dict,
                            target: np.ndarray, corpus: list, replace=False) -> np.array:
        container = []
        append = container.append
        length = len(target)
        begin = time()
        values = word_id.values()
        ls_value = list(values)
        ls = [corpus.count(i) for i in values]
        total = sum(ls)
        new = [i / total for i in ls]
        new_p = np.power(new, 0.75)
        new_p /= sum(new_p)
        for index, k in enumerate(target):
            print_result(index, length, begin)
            negative = np.array([k])
            while k in negative:
                negative = np.random.choice(ls_value,
                                            p=new_p,
                                            size=sample_size,
                                            replace=replace)
            append(negative)
        return np.array(container).reshape(len(container), sample_size)