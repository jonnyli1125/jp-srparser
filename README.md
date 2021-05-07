# Shift Reduce Dependency Parser for Japanese
## Overview
A neural network based dependency parser (syntax tree parser) for Japanese.

TL;DR Summary:
1. Implemented a shift reduce parser using the arc-standard transition system.
2. Implemented a neural network using PyTorch to predict the next action given a parser state.
3. Trained neural network with [UD Japanese GSD treebank](https://universaldependencies.org/treebanks/ja_gsd/index.html) and pretrained word embedding weights from [Wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/).

## Training
Run `python3 model.py` to execute training loop.

Model weights are saved in PyTorch model state format to `model.pth`. Other relevant files such as `model_lists.txt` and `embeddings/jawiki_gsd_word2vec.txt` are also required to load the model.

The model should generally converge at approximately .96 LAS (Labelled Attachment Score), .97 UAS (Unlabelled Attachment Score). Increasing the hyperparameters (embed size, hidden size) beyond the specified defaults may marginally improve accuracy.

## References
- http://www.cs.toronto.edu/~gpenn/csc485/a1.pdf
- http://www1.cs.columbia.edu/~stratos/research/sr_parsing.pdf  
- https://universaldependencies.org/treebanks/ja_gsd/index.html
- https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
