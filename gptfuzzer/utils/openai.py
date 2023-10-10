import openai
import logging


def openai_request(messages, model='gpt-3.5-turbo', temperature=1, top_n=1, max_trials=100, api_key=None):
    if api_key is None and openai.api_key is None:
        raise ValueError("OpenAI API key must be provided.")
    else:
        openai.api_key = api_key

    for _ in range(max_trials):
        try:
            results = openai.Completion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=top_n,
            )

            assert len(results['choices']) == top_n
            return results
        except Exception as e:
            logging.warning("OpenAI API call failed. Retrying...")
