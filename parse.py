from enum import Enum

class Transition(Enum):
    SHIFT = 0
    LEFT_ARC = 1
    RIGHT_ARC = 2

class ParseState:
    def __init__(self, sentence, root_word=None, root_tag="ROOT"):
        self.sentence = [(root_word, root_tag)] + sentence  # (word, tag)
        self.stack = [0]  # index of sentence
        self.next = 1  # index of sentence
        self.deps = set()  # (head, dependency, deprel)

    @property
    def complete(self):
        return len(self.stack) <= 1 and self.next >= len(self.sentence)

    def state(self):
        return (self.sentence, self.stack[:], self.next, frozenset(self.deps))

    def parse_transition(self, transition, deprel):
        if transition == Transition.SHIFT:
            if self.next >= len(self.sentence):
                raise ValueError("Buffer index out of range")
            self.stack.append(self.next)
            self.next += 1
        elif transition == Transition.LEFT_ARC:
            if len(self.stack) < 2:
                raise ValueError("Stack index out of range")
            self.deps.add((self.stack[-1], self.stack[-2], deprel))
            del self.stack[-2]
        elif transition == Transition.RIGHT_ARC:
            if len(self.stack) < 2:
                raise ValueError("Stack index out of range")
            self.deps.add((self.stack[-2], self.stack[-1], deprel))
            self.stack.pop()
        else:
            raise ValueError("Invalid transition")

    def get_oracle(self, oracle_deps):
        s = self.stack

        def nested_deps_added():
            for (j, k), l in oracle_deps.items():
                if s[-1] == j and (j, k, l) not in self.deps:
                    return False
            return True

        if len(s) < 2:
            return Transition.SHIFT, None
        elif (s[-1], s[-2]) in oracle_deps:
            return Transition.LEFT_ARC, oracle_deps[(s[-1], s[-2])]
        elif (s[-2], s[-1]) in oracle_deps and nested_deps_added():
            return Transition.RIGHT_ARC, oracle_deps[(s[-2], s[-1])]
        else:
            return Transition.SHIFT, None

def parse(self, sentence, model):
    """Parse sentence into dependencies using model."""
    state = ParseState(sentence)
    while not state.complete:
        transition, deprel = model.predict(state)  # TODO: implement
        state.parse_transition(transition, deprel)
    return state.deps
