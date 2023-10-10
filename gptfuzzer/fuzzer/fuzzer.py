import logging

from .mutator import Mutator, MutatePolicy
from .selection import SelectPolicy
from gptfuzzer.llm import LLM


class PromptNode:
    def __init__(self,
                 gptfuzzer: 'GPTFuzzer',
                 prompt: str,
                 response: str = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.gptfuzzer: 'GPTFuzzer' = gptfuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: list[PromptNode] = []
        if parent is None:
            self.level: int = 0
        else:
            self.level: int = parent.level + 1
            parent.child.append(self)

        self.index: int = len(self.gptfuzzer.prompt_nodes)
        self.gptfuzzer.prompt_nodes.append(self)


class GPTFuzzer:
    def __init__(self,
                 questions: 'list[str]',
                 targets: 'list[LLM]',
                 initial_seed: 'list[str]',
                 max_query: int = -1,
                 max_jailbreak: int = -1,
                 max_reject: int = -1,
                 max_iteration: int = -1,
                 ):

        self.questions: 'list[str]' = questions
        self.targets: 'list[LLM]' = targets
        self.prompt_nodes: 'list[PromptNode]' = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration

        self.selection_policy: SelectPolicy = None
        self.mutate_policy: MutatePolicy = None

    def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)

    def set_policy(self, selection_policy: SelectPolicy, mutate_policy: MutatePolicy):
        self.selection_policy = selection_policy
        self.mutate_policy = mutate_policy

    def run(self):
        logging.info("Fuzzing started!")

        while not self.is_stop():
            # seed selection
            seed = self.selection_policy.select()

            # mutation
            mutated_results = self.mutate_policy.mutate_single(seed)

            # attack

            # update

        logging.info("Fuzzing finished!")
