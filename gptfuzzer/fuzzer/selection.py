import random
import numpy as np

from gptfuzzer.fuzzer import GPTFuzzer, PromptNode


class SelectionPolicy:
    def __init__(self, fuzzer: 'GPTFuzzer'):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        raise NotImplementedError(
            "SelectionPolicy must implement select method.")

    def update(self, succ_num):
        pass


class RoundRobinSelectionPolicy(SelectionPolicy):
    def __init__(self, fuzzer: 'GPTFuzzer'):
        super().__init__(fuzzer)
        self.index: int = 0

    def select(self) -> PromptNode:
        seed = self.fuzzer.prompt_nodes[self.index]
        seed.visited_num += 1
        return seed

    def update(self, succ_num):
        self.index = (self.index - 1 + len(self.fuzzer.prompt_nodes)
                      ) % len(self.fuzzer.prompt_nodes)


class RandomSelectionPolicy(SelectionPolicy):
    def __init__(self, fuzzer: 'GPTFuzzer'):
        super().__init__(fuzzer)

    def select(self) -> PromptNode:
        seed = random.choice(self.fuzzer.prompt_nodes)
        seed.visited_num += 1
        return seed
