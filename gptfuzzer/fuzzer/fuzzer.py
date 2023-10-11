import logging

from .mutator import Mutator, MutatePolicy
from .selection import SelectPolicy

from gptfuzzer.llm import LLM
from gptfuzzer.utils.template import synthesis_message
from gptfuzzer.utils.predict import Predictor


class PromptNode:
    def __init__(self,
                 gptfuzzer: 'GPTFuzzer',
                 prompt: str,
                 response: str = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.gptfuzzer: 'GPTFuzzer' = gptfuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.results: 'list[int]' = results
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: list[PromptNode] = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child[index] = self

    @property
    def num_jailbreak(self):
        return sum(self.results)

    @property
    def num_reject(self):
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        return sum(self.results)


class GPTFuzzer:
    def __init__(self,
                 questions: 'list[str]',
                 target: LLM,
                 predictor: Predictor,
                 initial_seed: 'list[str]',
                 max_query: int = -1,
                 max_jailbreak: int = -1,
                 max_reject: int = -1,
                 max_iteration: int = -1,
                 ):

        self.questions: 'list[str]' = questions
        self.target: LLM = target
        self.predictor = predictor
        self.prompt_nodes: 'list[PromptNode]' = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()

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
            self.evluate(mutated_results)

            # update
            self.update(mutated_results)

        logging.info("Fuzzing finished!")

    def evluate(self, prompt_nodes: 'list[PromptNode]'):
        for prompt_node in prompt_nodes:
            responses = []
            for question in self.questions:
                message = synthesis_message(question, prompt_node.prompt)
                if message is None:  # The prompt is not valid
                    prompt_node.response = []
                    prompt_node.results = []
                    break

                responses.append(self.send(message))
            else:
                prompt_node.response = responses
                prompt_node.results = self.predictor.predict(
                    prompt_node.prompt, responses)

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.current_iteration += 1

        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)

            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

        self.selection_policy.update()
