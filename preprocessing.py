import re


def sub_space(string):
    return re.sub(r' {2,}', ' ', string)


def sub_initial_urdu(string):
    return re.sub(r'[۔()\'"”“،]', ' ', string)


with open('data/counter.txt') as file:
    text = file.read()

sentences = []
for sentence in sub_space(sub_initial_urdu(text)).split('\n'):
    sentences.append(sentence)

with open('data/counter_preprocessed.txt', 'w') as datafile:
    # store the data as binary data stream
    datafile.writelines("%s\n" % sentence for sentence in sentences)
