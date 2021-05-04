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
        return len(state.stack) <= 1 and state.next >= len(state.sentence)

    def state(self):
        return (self.sentence, self.stack, self.next, self.deps)

    def parse_transition(self, transition, deprel):
        if transition == Transition.SHIFT:
            if self.next >= len(self.sentence):
                raise ValueError("Buffer index out of range")
            self.stack.append(self.next)
            self.next += 1
        elif transition == Transition.LEFT_ARC:
            if len(self.stack) < 2:
                raise ValueError("Stack index out of range")
            self.deps.append((stack[-1], stack[-2], deprel))
            del stack[-2]
        elif transition == Transition.RIGHT_ARC:
            if len(self.stack) < 2:
                raise ValueError("Stack index out of range")
            self.deps.append((stack[-2], stack[-1], deprel))
            stack.pop()
        else:
            raise ValueError("Invalid transition")

    def get_oracle(self, oracle_deps):
        s = self.stack

        def nested_deps():
            for (i, j), l in oracle_deps.items():
                if i == s[-1] and (i, j, l) not in self.deps:
                    return True
            return False

        if len(s) < 2:
            return Transition.SHIFT, None
        elif (s[-1], s[-2]) in oracle_deps:
            return Transition.LEFT_ARC, oracle_deps[(s[-1], s[-2])]
        elif (s[-2], s[-1]) in oracle_deps and not nested_deps():
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
