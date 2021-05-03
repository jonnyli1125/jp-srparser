import pyconll

from parser import ParseState


TRAIN_PATH = 'corpora/ja_gsd-ud/ja_gsd-ud-train.conllu'
DEV_PATH = 'corpora/ja_gsd-ud/ja_gsd-ud-dev.conllu'
TEST_PATH = 'corpora/ja_gsd-ud/ja_gsd-ud-test.conllu'


def get_train_data():
    return pyconll.load_from_file(TRAIN_PATH)


def get_dev_data():
    return pyconll.load_from_file(DEV_PATH)


def get_test_data():
    return pyconll.load_from_file(TEST_PATH)


def get_n_leftmost_deps(deps, head_i, n):
    return sorted(dep for dep in deps if head_i == dep[0])[:n]


def get_n_rightmost_deps(deps, head_i, n):
    return sorted(dep for dep in deps if head_i == dep[0], reverse=True)[:n]


def get_oracle_deps(tree):
    """Given pyconll tree, return (head, dep) -> deprel dictionary."""
    res = {}
    q = [tree]
    while q:
        node = q.pop()
        for child in node:
            res[(node.data.id, child.data.id)] = child.data.deprel
            q.append(child)
    return res


def get_sentence(sentence):
    """Given pyconll sentence, return sentence list of (word, tag)."""
    return [(t.form, t.upos) for t in sentence]


def get_deps_transitions(sentence, oracle_deps):
    """Given sentence and oracle_deps, return deps and transition history."""
    state = ParseState(sentence)
    history = []
    while not state.complete:
        transition, deprel = state.get_oracle(oracle_deps)
        state.parse_transition(transition, deprel)
        history.append((state.state(), (transition, deprel)))
    assert state.deps == {(i, j, l) for (i, j), l in oracle_deps.items()}
    return state.deps, history
