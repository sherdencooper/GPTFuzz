import torch
import openai
from fastchat.model import load_model, get_conversation_template
import logging
import time

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):  #jiahao: add a function to overwrite later
        raise NotImplementedError("LLM must implement generate method.")

class LocalLLM(LLM):
    def __init__(self,
                 model_path,
                 device='cuda',
                 num_gpus=1,
                 max_gpu_memory=None,
                 dtype=torch.float16,
                 load_8bit=False,
                 cpu_offloading=False,
                 revision=None,
                 debug=False,
                 temperature=0.01,
                 repetition_penalty=1.0,
                 max_new_tokens=512
                 ):
        super().__init__()

        self.model, self.tokenizer = self.create_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )
        self.model_path = model_path
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
    @torch.inference_mode()
    def create_model(self, model_path,
                     device='cuda',
                     num_gpus=1,
                     max_gpu_memory=None,
                     dtype=torch.float16,
                     load_8bit=False,
                     cpu_offloading=False,
                     revision=None,
                     debug=False):
        model, tokenizer = load_model(
            model_path,
            device,
            num_gpus,
            max_gpu_memory,
            dtype,
            load_8bit,
            cpu_offloading,
            revision=revision,
            debug=debug,
        )

        return model, tokenizer

    @torch.inference_mode() #jiahao: Why the name is send?
    def generate(self, prompt):
        # FIXME: the function has some internal requirements  #jiahao: What does this mean?
        conv_temp = get_conversation_template(self.model_path)

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


class OnlineLLM(LLM):
    def __init__(self, 
                 model_path,
                 api_key,
                 temperature=0.0,
                 max_tokens=512,
                 max_trials=50,
                 failure_sleep_time=10,
                 n=1):
        super().__init__()

        if model_path in ['gpt-3.5-turbo', 'gpt-4']:
            self.generate = self.openai_generate
            openai.api_key = api_key
        
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n    # generate n responsess
        self.max_trials = max_trials
        self.failure_sleep_time = failure_sleep_time

    def openai_generate(self, prompt):
        responses = [" " for _ in range(self.n)]
        for _ in range(self.max_trials):
            try:
                results = openai.ChatCompletion.create(
                        model=self.model_path,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.temperature,
                        n = self.n
                    )
                assert len(results['choices']) == self.n
                responses =  [results['choices'][i]['message']['content'] for i in range(self.top_n)]
                break

            except Exception as e:
                logging.warning(f"OpenAI API call failed. Retrying... {e}")
                time.sleep(self.failure_sleep_time)
                continue

        return responses