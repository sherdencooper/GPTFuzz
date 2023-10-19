import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  # for debugging

from fastchat.model import add_model_args
import argparse
import pandas as pd
from gptfuzzer.fuzzer.selection import MCTSSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.llm import LocalLLM, OpenAILLM
from gptfuzzer.utils.predict import RoBERTaPredictor


def main(args):
    questions = pd.read_csv(args.seed_path)['text'].tolist()
    initial_seed = pd.read_csv(
        'datasets/prompts/GPTFuzzer.csv')['text'].tolist()

    # TODO: transfer other args into openai_model and target_model
    openai_model = OpenAILLM(args.model_path, args.api_key)
    target_model = LocalLLM(args.target_model)

    roberta_model = RoBERTaPredictor('hubert233/GPTFuzz')

    fuzzer = GPTFuzzer(
        questions=questions,
        target=target_model,
        predictor=roberta_model,
        initial_seed=initial_seed,
        mutate_policy=MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(openai_model),
            OpenAIMutatorExpand(openai_model),
            OpenAIMutatorGenerateSimilar(openai_model),
            OpenAIMutatorRephrase(openai_model),
            OpenAIMutatorShorten(openai_model)],
        ),
        select_policy=MCTSSelectPolicy(),
        energy=args.energy,
    )

    fuzzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--api_key', type=str, default='', help='API Key')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo',
                        help='openai model or open-sourced LLMs')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=500,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed_path", type=str,
                        default="datasets/prompts/GPTFuzzer.csv")
    add_model_args(parser)

    args = parser.parse_args()
    main(args)
