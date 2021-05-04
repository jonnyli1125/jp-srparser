# Shift Reduce Dependency Parser for Japanese
## Overview
A neural network based dependency parser (syntax tree parser) for Japanese.

TL;DR Summary:
1. Implemented a shift reduce parser using the arc-standard transition system.
2. Implemented a neural network using PyTorch to predict the next action given a parser state.
3. Trained neural network with [UD Japanese GSD treebank](https://universaldependencies.org/treebanks/ja_gsd/index.html) and pretrained word embedding weights from [Wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/).

## Training
Run `python3 model.py` to execute training loop.  
Model is saved in PyTorch model state format to `model.pth`.  
The model should be able to achieve above .98 LAS (Labelled Attachment Score), .99 UAS (Unlabelled Attachment Score) with default hyperparameters.

## References
- http://www.cs.toronto.edu/~gpenn/csc485/a1.pdf
- http://www1.cs.columbia.edu/~stratos/research/sr_parsing.pdf  
- https://universaldependencies.org/treebanks/ja_gsd/index.html
- https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
