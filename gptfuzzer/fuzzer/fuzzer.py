
class PromptNode:
    def __init__(self,
                 gptfuzzer: 'GPTFuzzer',
                 prompt: str,
                 response: str = None,
                 parent: 'PromptNode' = None):
        self.gptfuzzer: 'GPTFuzzer' = gptfuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.child: list[PromptNode] = []
        if parent is None:
            self.level: int = 0
        else:
            self.level: int = parent.level + 1
            parent.child.append(self)

        self.index: int = len(self.gptfuzzer.prompt_nodes)
        self.gptfuzzer.prompt_nodes.append(self)


class GPTFuzzer:
    def __init__(self):
        self.prompt_nodes: list[PromptNode] = []
