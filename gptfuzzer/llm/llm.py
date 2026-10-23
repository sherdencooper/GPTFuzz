import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
from vllm import LLM as vllm
from vllm import SamplingParams
from google import genai  # New Google Gen AI SDK (google-genai package)
from google.genai import types
from anthropic import Anthropic

from gptfuzzer.orcarouter import OrcaRouterLLM as _OrcaRouterLLM


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

        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

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
            output_ids = output_ids[:, len(input_ids[0]):]
            outputs.extend(self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
        return outputs


class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.95,
                 system_message=None
                 ):
        super().__init__()
        self.model_path = model_path
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

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

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs


class BardLLM(LLM):
    def generate(self, prompt):
        return

class PaLM2LLM(LLM):
    """DEPRECATED: PaLM API was decommissioned on August 15, 2024.

    Please migrate to GeminiLLM instead.
    See PALM_TO_GEMINI_MIGRATION.md for detailed migration guide.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise DeprecationWarning(
            "PaLM API was decommissioned on August 15, 2024. "
            "Use GeminiLLM instead. See PALM_TO_GEMINI_MIGRATION.md for migration guide."
        )

    def generate(self, *args, **kwargs):
        raise DeprecationWarning("PaLM API is no longer available. Use GeminiLLM.")

    def generate_batch(self, *args, **kwargs):
        raise DeprecationWarning("PaLM API is no longer available. Use GeminiLLM.")


class GeminiLLM(LLM):
    """Google Gemini API implementation for GPTFuzzer.

    Replacement for deprecated PaLM2LLM. Supports Gemini 1.5 and 2.0 models.
    Uses the new google-genai SDK (Client-based API).
    """

    def __init__(self,
                 model_path='gemini-2.5-flash',
                 api_key=None,
                 system_message=None,
                 safety_settings=None
                ):
        """Initialize Gemini LLM.

        Args:
            model_path: Model name (e.g., 'gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-pro')
            api_key: Google AI API key from https://aistudio.google.com/apikey
            system_message: System instruction for the model (optional)
            safety_settings: Optional safety settings list (e.g., ['BLOCK_NONE', 'BLOCK_NONE', ...])
        """
        super().__init__()

        if not api_key:
            raise ValueError(
                'Gemini API key is required. Get one at https://aistudio.google.com/apikey'
            )

        # Create client with API key
        self.client = genai.Client(api_key=api_key)
        self.model_path = model_path
        self.system_message = system_message

        # Store safety settings for config (new API uses simpler string-based settings)
        # Default to permissive settings for research purposes
        self.safety_settings = safety_settings if safety_settings else ['BLOCK_NONE'] * 4

    def __del__(self):
        """Cleanup client connection."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except:
                pass

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        """Generate responses using Gemini API (new google-genai SDK).

        Args:
            prompt: Input prompt string
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum output tokens
            n: Number of candidates to generate
            max_trials: Maximum retry attempts on failure
            failure_sleep_time: Sleep duration between retries (seconds)

        Returns:
            List of generated text strings

        Note:
            Gemini API may block responses based on safety filters.
            Blocked responses return empty string " ".
        """
        # Create generation config using new API
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=0.95,
            top_k=40,
            system_instruction=self.system_message if self.system_message else None,
        )

        results = []

        # Generate n responses (make multiple calls if n > 1)
        for _ in range(n):
            for trial in range(max_trials):
                try:
                    # Use new client-based API
                    response = self.client.models.generate_content(
                        model=self.model_path,
                        contents=prompt,
                        config=config
                    )

                    # Extract text from response
                    if hasattr(response, 'text') and response.text:
                        results.append(response.text)
                    elif hasattr(response, 'candidates') and response.candidates:
                        # Fallback: try to extract from candidates
                        candidate_text = str(response.candidates[0]) if response.candidates else " "
                        results.append(candidate_text if candidate_text else " ")
                    else:
                        # Response blocked or empty
                        logging.warning(f"Gemini response blocked or empty")
                        results.append(" ")

                    break  # Success, exit retry loop

                except Exception as e:
                    logging.warning(
                        f"Gemini API call failed due to {e}. "
                        f"Retrying {trial+1} / {max_trials} times..."
                    )

                    if trial == max_trials - 1:
                        # Max retries reached, return empty response
                        results.append(" ")
                    else:
                        time.sleep(failure_sleep_time)

        return results[:n]  # Ensure we return exactly n results

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1,
                      max_trials=10, failure_sleep_time=5):
        """Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of input prompt strings
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            n: Number of candidates per prompt
            max_trials: Maximum retry attempts
            failure_sleep_time: Sleep duration between retries

        Returns:
            List of generated text strings (flattened)
        """
        results = []

        # Use thread pool for concurrent API calls (limit to 5 to avoid rate limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    self.generate,
                    prompt,
                    temperature,
                    max_tokens,
                    n,
                    max_trials,
                    failure_sleep_time
                ): prompt
                for prompt in prompts
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    logging.error(f"Batch generation failed: {e}")
                    results.extend([" " for _ in range(n)])

        return results


class ClaudeLLM(LLM):
    def __init__(self,
                 model_path='claude-haiku-4-5-20251001',
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key:
            raise ValueError('Claude API key is required')

        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."
        self.anthropic = Anthropic(api_key=api_key)

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        for _ in range(n):
            for trial in range(max_trials):
                try:
                    response = self.anthropic.messages.create(
                        model=self.model_path,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=self.system_message,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = next((block.text for block in response.content if hasattr(block, 'text')), " ")
                    results.append(text)
                    break
                except Exception as e:
                    logging.warning(
                        f"Claude API call failed due to {e}. Retrying {trial+1} / {max_trials} times...")
                    if trial == max_trials - 1:
                        results.append(" ")
                    else:
                        time.sleep(failure_sleep_time)
        return results

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        # if model_path not in ['gpt-3.5-turbo', 'gpt-4']:
        #     raise ValueError(
        #         'OpenAI model path should be gpt-3.5-turbo or gpt-4')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results


class OrcaRouterLLM(_OrcaRouterLLM, LLM):
    """OrcaRouter Cyber Inference provider for mutate and target models.

    The implementation lives in :mod:`gptfuzzer.orcarouter` so that the
    credential, catalog and transport layers stay importable without the
    optional GPU dependencies.  This subclass only adds the project's ``LLM``
    base class for type compatibility.
    """
