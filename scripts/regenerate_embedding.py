from itertools import chain
from gensim.models.keyedvectors import KeyedVectors
from data import get_train_data, get_dev_data, get_test_data


orig_path = 'embeddings/jawiki_20180420_100d.txt'
out_path = 'embeddings/jawiki_gsd_word2vec.txt'

# The pretrained Japanese wiki word2vec model is very large, so we will run this
# script to remove all words that aren't in the GSD corpus.
model = KeyedVectors.load_word2vec_format(orig_path)
train, dev, test = get_train_data(), get_dev_data(), get_test_data()
visited = set()
gsd_vocab = []
gsd_vectors = []
count = 0
for s in chain(train, dev, test):
    for t in s:
        try:
            if t.form in visited:
                continue
            visited.add(t.form)
            gsd_vectors.append(model.get_vector(t.form))
            gsd_vocab.append(t.form)
        except KeyError:
            pass
gsd_model = KeyedVectors(vector_size=model.vectors.shape[1])
gsd_model.add_vectors(gsd_vocab, gsd_vectors)
gsd_model.save_word2vec_format(out_path)
