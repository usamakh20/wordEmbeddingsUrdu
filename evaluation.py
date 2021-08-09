# Adapted from https://github.com/k-kawakami/embedding-evaluation/blob/master/wordsim/wordsim.py
import pickle
import numpy as np
from ngrams_urdu import x_fun
from scipy import linalg, stats

data_dir = 'data/'


def load_evaluation_data(filename, score_column, header):
    evaluation_list = []
    for index, line in enumerate(open(data_dir + filename)):
        if (not header or index > 0) or not header:  # if header then index > 0 else continue
            evaluation_list.append(
                [float(w) if i == score_column else w for i, w in enumerate(line.strip().split('\t'))])
    return evaluation_list


def load_ngrams(filename):
    return pickle.load(open(data_dir + filename, "rb"))


def load_vectors(filename):
    f = open(data_dir + filename, "r")
    word2vec = {}
    for wn, line in enumerate(f):
        if wn > 0:
            line = line.lower().strip()
            word = line.split()[0]
            word2vec[word] = np.array(list(map(float, line.split()[1:])))
    return word2vec


def cos(vec1, vec2):
    return vec1.dot(vec2) / (linalg.norm(vec1) * linalg.norm(vec2))


def rho(vec1, vec2):
    return stats.stats.spearmanr(vec1, vec2)[0]


def evaluate(word_dict, eval_list, score_column, word_vector=True):
    vocab = word_dict.keys()
    pred, label, found, notfound = [], [], 0, 0
    for datum in eval_list:
        if word_vector:
            if datum[0] in vocab and datum[1] in vocab:
                found += 1
                pred.append(cos(word_dict[datum[0]], word_dict[datum[1]]))
                label.append(datum[score_column])
            else:
                notfound += 1
        else:
            if datum[1] in word_dict[datum[0]]:
                found += 1
                pred.append(word_dict[datum[0]][datum[1]])
                label.append(datum[score_column])
            else:
                notfound += 1

    return found, notfound, rho(pred, label)


if __name__ == '__main__':
    evaluation_data_wordsim = load_evaluation_data('wordsim_similarity_goldstandard_urdu.txt', 2, False)
    evaluation_data_simlex = load_evaluation_data('SimLex-999_urdu.txt', 3, True)
    fasttext_vectors = load_vectors('fasttext_urdu.vec')
    ngram_dict = load_ngrams('ngram_urdu.pickle')
    print('Fasttext on WordSim Gold Standard: ' + str(evaluate(fasttext_vectors, evaluation_data_wordsim, 2)))
    print('Fasttext on SimLex-999: ' + str(evaluate(fasttext_vectors, evaluation_data_simlex, 3)))
    print('bigram on WordSim Gold Standard: ' + str(evaluate(ngram_dict, evaluation_data_wordsim, 2, False)))
    print('bigram on SimLex-999: ' + str(evaluate(ngram_dict, evaluation_data_simlex, 3, False)))
