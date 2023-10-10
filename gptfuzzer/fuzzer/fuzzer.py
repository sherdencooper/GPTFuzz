class PromptNode:
    def __init__(self,
                 gptfuzzer: 'GPTFuzzer',
                 prompt: str,
                 response: str = None,
                 parent: 'PromptNode' = None):
        self.gptfuzzer = gptfuzzer
        self.prompt = prompt
        self.response = response

        self.parent = parent
        self.child = []
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1
            parent.child.append(self)

        self.index = len(self.gptfuzzer.prompt_nodes)
        self.gptfuzzer.prompt_nodes.append(self)

class GPTFuzzer:
    def __init__(self):
        self.prompt_nodes = []
