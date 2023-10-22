import torch
import openai
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
from vllm import LLM as vllm
from vllm import SamplingParams

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
                 system_message=None
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
        
        # monkey patch for latest FastChat to use llama2's official system message
        if 'Llama-2' in model_path and system_message is None:   
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
            "If you don't know the answer to a question, please don't share false information."
        elif system_message is None:
            self.system_message = system_message
        else:
            self.system_message = None

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

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)
        
    @torch.inference_mode() 
    def generate(self, prompt, temperature=0.01, max_tokens=512, repetition_penalty=1.0):
        conv_temp = get_conversation_template(self.model_path)
        self.set_system_message(conv_temp)
        
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
        

    @torch.inference_mode()
    def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=16):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)
            
            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)
            
            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)
            
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
        # load the input_ids batch by batch to avoid OOM
        outputs = []
        for i in range(0, len(input_ids), batch_size):
            output_ids = self.model.generate(
                torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
                do_sample=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
            output_ids = output_ids[:, len(input_ids[0]) :]
            outputs.extend(self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs

class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.95,
                 system_message=None
                 ):
        super().__init__()
        self.model_path = model_path
        self.model = vllm(self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        # monkey patch for latest FastChat to use llama2's official system message
        if 'Llama-2' in model_path and system_message is None:   
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
            "If you don't know the answer to a question, please don't share false information."
        elif system_message is None:
            self.system_message = system_message
        else:
            self.system_message = None
            
    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)
    
    def generate(self, prompt, temperature=0, max_tokens=512):
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)
        
            
    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)
            
            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)
            
            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)       
                
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
        results = self.model.generate(prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs
        
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
            raise ValueError(
                'OpenAI model path should be gpt-3.5-turbo or gpt-4')
        openai.api_key = api_key
        self.model_path = model_path

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, request_timeout=20, max_trials=10, failure_sleep_time=5):
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
                    n=n,
                    request_timeout=request_timeout
                )
                return [results['choices'][i]['message']['content'] for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]
    
    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, request_timeout=20, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n, request_timeout, max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                    results.extend(future.result())
        return results
