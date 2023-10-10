import openai
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import csv
import os
import subprocess
import numpy as np
from fuzz_utils import *
from llm_utils.creat_model import prepare_model_and_tok
import random
import argparse
import copy
from fastchat.model import add_model_args
from generate_response import replace_template, execute_query

def main(args, MODEL, TOK, questions_train, questions_test, attack_prompt):
    train_asr = np.array(attack_prompt['response'].tolist())
    prompts = np.array(attack_prompt['text'].tolist())
    top_5_asr_index = np.argsort(train_asr)[::-1][:5]
    top_5_prompts = prompts[top_5_asr_index]

    for question in tqdm(questions_test):
        for index, prompt in enumerate(top_5_prompts):
            replaced_prompt = replace_template(question, prompt)
            response = execute_query(args, replaced_prompt, MODEL, TOK)
            print(response)

    for question in tqdm(questions_train):
        for index, prompt in enumerate(top_5_prompts):
            replaced_prompt = replace_template(question, prompt)
            response = execute_query(args, replaced_prompt, MODEL, TOK)
            print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--openai_key', type=str, default='You must have an OpenAI key', help='OpenAI key')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='openai model or open-sourced LLMs')
    parser.add_argument("--temperature", type=float, default=0.01)          
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--mode", type=str, default='multi-single')
    parser.add_argument("--prompt_path", type=str, default='./datasets/prompts_generated/multi_single/multi_single_Llama-2-7b-chat-hf.csv')
    add_model_args(parser)
    args = parser.parse_args()
    args.num_gpus = 1
    MODEL, TOK = prepare_model_and_tok(args)
    questions_test = pd.read_csv('datasets/questions/question_list.csv')['text'].tolist()[20:]  #use 80 questions as the evaluate set
    questions_train = pd.read_csv('datasets/questions/question_list.csv')['text'].tolist()[:20]  #use 20 questions as the train set
    attack_prompt = pd.read_csv(args.prompt_path)
    print(args)
    main(args, MODEL, TOK, questions_train, questions_test, attack_prompt)
