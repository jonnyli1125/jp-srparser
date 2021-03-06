from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (get_word2vec, get_train_data, get_dev_data, get_test_data,
                  get_word_tag_deprel_lists, load_word_tag_deprel_lists,
                  save_word_tag_deprel_lists, Encoder, CorpusDataset,
                  MODEL_LISTS_PATH)


class ParserModel(nn.Module):
    def __init__(self, encoder=None, word2vec=None, dropout=0.5,
                 hidden_size=100):
        super().__init__()

        if not word2vec:
            word2vec = get_word2vec()
        if not encoder:
            encoder = Encoder(*load_word_tag_deprel_lists(MODEL_LISTS_PATH))
        n_tag_ids = len(encoder.id2tag) + 1
        n_deprel_ids = len(encoder.id2deprel) + 1
        n_word_features = Encoder.n_word_features
        n_tag_features = Encoder.n_tag_features
        n_deprel_features = Encoder.n_deprel_features
        n_classes = encoder.n_classes
        embed_size = word2vec.vector_size
        self.encoder = encoder

        # init embeddings
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

    def get_concat_embedding(self, X_word, X_tag, X_deprel):
        X_word_embed = self.word_embedding(X_word)
        X_tag_embed = self.tag_embedding(X_tag)
        X_deprel_embed = self.deprel_embedding(X_deprel)
        X_concat_embed = torch.cat((X_word_embed, X_tag_embed, X_deprel_embed), 1)
        B = X_word.shape[0]
        return X_concat_embed.reshape((B, -1)).float()

    def forward(self, X_word, X_tag, X_deprel):
        X_word, X_tag, X_deprel = X_word.int(), X_tag.int(), X_deprel.int()
        X = self.get_concat_embedding(X_word, X_tag, X_deprel)
        return self.linear_stack(X)

    def predict(self, state):
        X_word, X_tag, X_deprel = self.encoder.encode_state(*state)
        with torch.no_grad():
            logits = self(torch.tensor(X_word).reshape(1, -1),
                          torch.tensor(X_tag).reshape(1, -1),
                          torch.tensor(X_deprel).reshape(1, -1))
            return self.encoder.decode_target(logits)
        raise ValueError("Unable to predict transition.")


def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    n_batches = len(dataloader)
    for i, (X_word, X_tag, X_deprel, y_l, y_ul) in enumerate(dataloader):
        # compute prediction and get loss
        pred = model(X_word, X_tag, X_deprel)
        loss = loss_fn(pred, y_l.argmax(1))
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("[Batch {}/{}] Loss: {}".format(i+1, n_batches, loss.item()))


def train(train_dataset, dev_dataset, model, batch_size=2048, n_epochs=10,
          lr=0.001):
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dev_dl = DataLoader(dev_dataset, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for i in range(n_epochs):
        print("Begin epoch {}/{}...".format(i+1, n_epochs))
        train_epoch(train_dl, model, loss_fn, optimizer)
        las, uas = evaluate(dev_dl, model)
        print("[Epoch {}/{}] LAS: {}, UAS: {}".format(i+1, n_epochs, las, uas))


def evaluate(dataloader, model):
    """
    Evaluate model on dataset and return (LAS, UAS).

    LAS = Labelled attachment score, i.e. acc of predicting transition + deprel
    UAS = Unlabelled attachment score, i.e. acc of predicting transition
    """
    size = len(dataloader.dataset)
    model.eval()
    las, uas = 0, 0
    with torch.no_grad():
        for X_word, X_tag, X_deprel, y_l, y_ul in dataloader:
            pred = model(X_word, X_tag, X_deprel)
            pred_onehot = F.one_hot(pred.argmax(1), y_l.shape[1])
            las += (pred_onehot * y_l).sum().item()
            uas += (pred_onehot * y_ul).sum().item()
    las /= size
    uas /= size
    return las, uas


def main():
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print('Running on GPU: {}.'.format(torch.cuda.get_device_name()))
    else:
        print('Running on CPU.')
    print("Loading corpus and pretrained word embedding...")
    word2vec = get_word2vec()
    train_data = get_train_data()
    dev_data = get_dev_data()
    test_data = get_test_data()
    all = chain(train_data, dev_data, test_data)
    word_list, tag_list, deprel_list = get_word_tag_deprel_lists(word2vec, all)
    print("Corpus and embedding loaded. {} words, {} tags, {} deprels".format(
        len(word_list), len(tag_list), len(deprel_list)))
    save_word_tag_deprel_lists(
        MODEL_LISTS_PATH, word_list, tag_list, deprel_list)
    print("Saved word/tag/deprel lists to {}.".format(MODEL_LISTS_PATH))

    encoder = Encoder(word_list, tag_list, deprel_list)
    print("Generating datasets from corpus...")
    train_dataset = CorpusDataset(train_data, encoder)
    dev_dataset = CorpusDataset(dev_data, encoder)
    test_dataset = CorpusDataset(test_data, encoder)
    print("Datasets generated. Train: n={}, Dev: n={}, Test: n={}".format(
        len(train_dataset), len(dev_dataset), len(test_dataset)))

    model = ParserModel(encoder, word2vec)
    print("Begin training...")
    train(train_dataset, dev_dataset, model)
    print("Training finished.")
    test_las, test_uas = evaluate(DataLoader(test_dataset), model)
    print("Test LAS: {}, UAS: {}".format(test_las, test_uas))
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch model state to model.pth")


if __name__ == '__main__':
    main()
