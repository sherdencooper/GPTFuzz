import torch
import openai
from fastchat.model import load_model, get_conversation_template
import logging
import time

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt): 
        raise NotImplementedError("LLM must implement generate method.")
    
    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")

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
    def generate(self, prompt, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        # FIXME: the function has some internal requirements  #jiahao: What does this mean?
        conv_temp = get_conversation_template(self.model_path)

        conv_temp.append_message(conv_temp.roles[0], prompt)
        conv_temp.append_message(conv_temp.roles[1], None)

        prompt_input = conv_temp.get_prompt()
        input_ids = self.tokenizer([prompt_input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens
        )

        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]):]

        return self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

class BardLLM(LLM):
    def generate(self, prompt):
        return 

class ClaudeLLM(LLM):
    def generate(self, prompt):
        return 

class OpenAILLM(LLM):
    def __init__(self, 
                 model_path,
                 api_key):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        if model_path not in ['gpt-3.5-turbo', 'gpt-4']:
            raise ValueError('OpenAI model path should be gpt-3.5-turbo or gpt-4')
        openai.api_key = api_key
        self.model_path = model_path
    
    def generate(self, prompt, temperature=0.0, max_tokens=512, n=1, request_timeout=20, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = openai.ChatCompletion.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n = n,
                    request_timeout=request_timeout
                )
                return [results['choices'][i]['message']['content'] for i in range(n)]
            except Exception as e:
                logging.warning(f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)
                continue
        
        return [" " for _ in range(n)] # return empty string if failed after max_trials
