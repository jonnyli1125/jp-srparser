import pyconll
from gensim.models.keyedvectors import KeyedVectors

from parser import ParseState, Transition


TRAIN_PATH = 'corpora/ja_gsd-ud/ja_gsd-ud-train.conllu'
DEV_PATH = 'corpora/ja_gsd-ud/ja_gsd-ud-dev.conllu'
TEST_PATH = 'corpora/ja_gsd-ud/ja_gsd-ud-test.conllu'
WORD2VEC_PATH = 'embeddings/jawiki_gsd_word2vec.txt'


def get_train_data():
    return pyconll.load_from_file(TRAIN_PATH)


def get_dev_data():
    return pyconll.load_from_file(DEV_PATH)


def get_test_data():
    return pyconll.load_from_file(TEST_PATH)


def get_word2vec():
    return KeyedVectors.load_word2vec_format(WORD2VEC_PATH)


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


def get_word_tag_deprel_lists(word2vec, conll):
    """Given KeyedVectors, conll, return word_list, tag_list, deprel_list."""
    word_list = word2vec.index_to_key
    tag_list = set()
    deprel_list = set()
    for sentence in conll:
        for t in sentence:
            tag_list.add(t.upos)
            deprel_list.add(t.deprel)
    return word_list, sorted(tag_list), sorted(deprel_list)


class Encoder:
    root_word = None
    root_tag = "ROOT"
    unk_word = "_"
    unk_tag = "_"
    unk_deprel = "_"

    def __init__(self, word_list, tag_list, deprel_list):
        self.id2word = [root_word] + word_list + [unk_word]
        self.id2tag = [root_tag] + tag_list + [unk_tag]
        # deprel list already includes root deprel
        self.id2deprel = deprel_list + [unk_deprel]
        self.word2id = {word: i for (i, word) in enumerate(self.id2word)}
        self.tag2id = {tag: i for (i, tag) in enumerate(self.id2tag)}
        self.deprel2id = {dr: i for (i, dr) in enumerate(self.id2deprel)}
        self.unk_word_id = len(self.id2word) - 1
        self.unk_tag_id = len(self.id2tag) - 1
        self.unk_deprel_id = len(self.id2deprel) - 1
        self.null_word_id = len(self.id2word)
        self.null_tag_id = len(self.id2tag)
        self.null_deprel_id = len(self.id2deprel)

    def encode_state(self, sentence, stack, next, deps):
        """
        Feature vector format:

        word/tag vectors (18 each):
            - top 3 ids on stack
            - top 3 ids on buffer
            - 1st and 2nd leftmost and rightmost dependants from top
              two words on stack (8)
            - leftmost-leftmost and rightmost-rightmost of top two words
              on stack (4)

        deprel vector (12):
            - 1st and 2nd leftmost and rightmost dependants from top
              two words on stack (8)
            - leftmost-leftmost and rightmost-rightmost of top two words
              on stack (4)
        """
        def word2id(word):
            return self.word2id.get(word, self.unk_word_id)

        def tag2id(tag):
            return self.tag2id.get(tag, self.unk_tag_id)

        def deprel2id(deprel):
            return self.deprel2id.get(deprel, self.unk_deprel_id)

        word_ids = np.ones(18) * self.null_word_id
        tag_ids = np.ones(18) * self.null_tag_id
        deprel_ids = np.ones(12) * self.null_deprel_id
        for i in range(min(len(stack), 3)):
            stack_word, stack_tag = sentence[stack[-1-i]]
            word_ids[i] = word2id(stack_word)
            tag_ids[i] = tag2id(stack_tag)
        for i in range(min(len(sentence) - next, 3)):
            buffer_word, buffer_tag = sentence[next+i]
            word_ids[3+i] = word2id(buffer_word)
            tag_ids[3+i] = tag2id(buffer_tag)
        for i in range(min(len(stack)), 2):
            for j, dep in get_n_leftmost_deps(deps, stack[-1-i], 2):
                dep_word, dep_tag = sentence[dep[1]]
                deprel = dep[2]
                word_ids[6+6*i+j] = word2id(dep_word)
                tag_ids[6+6*i+j] = tag2id(dep_tag)
                deprel_ids[6*i+j] = deprel2id(deprel)
                if j == 0:
                    for k, inner_dep in get_n_leftmost_deps(deps, dep, 1):
                        inner_dep_word, inner_dep_tag = sentence[inner_dep[1]]
                        inner_deprel = inner_dep[2]
                        word_ids[10+6*i] = word2id(inner_dep_word)
                        tag_ids[10+6*i] = tag2id(inner_dep_tag)
                        deprel_ids[4+6*i] = deprel2id(inner_deprel)
            for j, dep in get_n_rightmost_deps(deps, stack[-1-i], 2):
                dep_word, dep_tag = sentence[dep[1]]
                deprel = dep[2]
                word_ids[8+6*i+j] = word2id(dep_word)
                tag_ids[8+6*i+j] = tag2id(dep_tag)
                deprel_ids[2+6*i+j] = deprel2id(deprel)
                if j == 0:
                    for k, inner_dep in get_n_rightmost_deps(deps, dep, 1):
                        inner_dep_word, inner_dep_tag = sentence[inner_dep[1]]
                        inner_deprel = inner_dep[2]
                        word_ids[11+6*i] = word2id(inner_dep_word)
                        tag_ids[11+6*i] = tag2id(inner_dep_tag)
                        deprel_ids[5+6*i] = deprel2id(inner_deprel)
        return word_ids, tag_ids, deprel_ids

    def encode_target(self, transition, deprel):
        """
        Transition + deprel vector format:

        one-hot encoded vector, where indexes represent:
            - 0: SHIFT
            - [1, n+1): LEFT_ARC, deprel 1+i
            - [n+1, 2n+1): RIGHT_ARC, deprel n+1+i
        where n = len(id2deprel), 0 <= i < n
        """
        n = len(self.id2deprel)
        res = np.zeros(2*n+1)
        if transition == Transition.SHIFT:
            res[0] = 1
        else:
            i = self.deprel2id.get(deprel, self.unk_deprel_id)
            if transition == Transition.LEFT_ARC:
                res[1+i] = 1
            else:
                res[n+1+i] = 1
        return res

    def encode_pair(self, pair):
        return self.encode_state(*pair[0]), self.encode_target(*pair[1])