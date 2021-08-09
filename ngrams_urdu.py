import re
import pickle
from nltk import bigrams
from collections import defaultdict


def x_fun():
    return defaultdict(int)


if __name__ == '__main__':

    with open('data/counter_preprocessed.txt', 'r') as datafile:
        sentences = datafile.read()

    preprocessed_sentences = [sentence.split() for sentence in re.sub(r'\n', ' ', sentences).split('۔')]

    # Create a placeholder for model
    model = defaultdict(x_fun)

    # Count frequency of co-occurrence
    for sentence in preprocessed_sentences:
        for w1, w2, in bigrams(sentence, pad_right=True, pad_left=True):
            model[w1][w2] += 1

    # Let's transform the counts to probabilities
    for w1 in model:
        total_count = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_count

    with open('data/ngram_urdu.pickle', 'wb') as file:
        pickle.dump(model, file)

    print(dict(model['دنیا']))
