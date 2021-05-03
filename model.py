import numpy as np
import torch
import torch.nn as nn


class ParserModel(nn.Module):
    def __init__(self, word2vec, n_word_ids, n_tag_ids, n_deprel_ids,
                 n_word_features, n_tag_features, n_deprel_features, n_classes,
                 embed_size=None, dropout=0.5, hidden_size=200):
        super().__init__()
        # init embeddings
        if not embed_size:
            embed_size = word2vec.vector_size
        self.word_embedding = self.init_word_embedding(word2vec)
        self.tag_embedding = nn.Embedding(n_tag_ids, embed_size)
        self.deprel_embedding = nn.Embedding(n_deprel_ids, embed_size)
        # init layers
        N = n_word_features + n_tag_features + n_deprel_features
        self.linear_stack = nn.Sequential(
            nn.Linear(N * embed_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
            # cross entropy loss already applies softmax to output
        )

    def init_word_embedding(self, word2vec):
        # Add root, unk, null vectors
        root = np.random.uniform(-.01, .01, word2vec.vector_size)
        unk = np.random.uniform(-.01, .01, word2vec.vector_size)
        null = np.random.uniform(-.01, .01, word2vec.vector_size)
        weights = torch.tensor(np.vstack((root, word2vec.vectors, unk, null)))
        return nn.Embedding.from_pretrained(weights, freeze=False)

    def get_concat_embedding(self, word_id_batch, tag_id_batch,
                             deprel_id_batch):
        word_batch = self.word_embedding(word_id_batch)
        tag_batch = self.tag_embedding(tag_id_batch)
        deprel_batch = self.deprel_embedding(deprel_id_batch)
        concat_batch = torch.cat((word_batch, tag_batch, deprel_batch), 1)
        B = word_id_batch.shape[0]
        return concat_batch.reshape((B, -1))

    def forward(self, word_id_batch, tag_id_batch, deprel_id_batch):
        x = self.get_concat_embedding(
            word_id_batch, tag_id_batch, deprel_id_batch)
        return self.linear_stack(x)
