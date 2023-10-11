import torch
import openai
from fastchat.model import load_model, get_conversation_template

from gptfuzzer.fuzzer import GPTFuzzer


class LLM:
    def __init__(self,
                 fuzzer: GPTFuzzer
                 ):
        self.fuzzer: GPTFuzzer = fuzzer
        self.model = None
        self.tokenizer = None


class LocalLLM(LLM):
    def __init__(self,
                 fuzzer: GPTFuzzer,
                 path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=0.5,
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 temperature=0.01,
                 repetition_penalty=1.0,
                 max_new_tokens=512
                 ):
        super().__init__(fuzzer)

        self.model, self.tokenizer = load_model(
            path,
            device,
            num_gpus,
            max_gpu_memory,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.path = path
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def create_model(self, path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=0.5,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        model, tokenizer = load_model(
            path,
            device,
            num_gpus,
            max_gpu_memory,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )

        return model, tokenizer

    @torch.inference_mode()
    def send(self, prompt):
        # FIXME: the function has some internal requirements
        conv_temp = get_conversation_template(self.path)

        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)

        prompt_input = conv_temp.get_prompt()
        input_ids = self.tokenizer([prompt_input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_new_tokens,
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
