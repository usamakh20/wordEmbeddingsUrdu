import os
import fasttext

if os.path.isfile('data/fasttext_urdu.bin'):
    model = fasttext.load_model("data/fasttext_urdu.bin")
else:
    model = fasttext.train_unsupervised('data/counter_preprocessed.txt')

print(model.get_word_vector("نقل"))

print(model.get_nearest_neighbors('نقل'))

if not os.path.isfile('data/fasttext_urdu.bin'):
    model.save_model("data/fasttext_urdu.bin")
