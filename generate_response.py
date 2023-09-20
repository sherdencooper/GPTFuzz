# some codes are adpated from https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/huggingface_api.py

import openai
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import csv
import os
import subprocess
import numpy as np
import argparse
import torch
from fastchat.model import add_model_args
from llm_utils.creat_model import prepare_model_and_tok
from llm_utils.model_inference import LLM_response


def replace_template(test_question, prompt):
    if '[INSERT PROMPT HERE]' in prompt:
        jailbreak_input = prompt.replace('[INSERT PROMPT HERE]', test_question)
        return jailbreak_input
    else:
        return False

def execute_query(args, mutant, MODEL, TOK):
    if 'gpt' in args.model_path:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": mutant},
            ],
            temperature=0,
        )
        return response['choices'][0]['message']['content']
    
    else:
        return LLM_response(args, MODEL, TOK, args.model_path, mutant)
    

def main(args, MODEL, TOK, initial_seed, questions):
    model_name = args.model_path.split('/')[-1]
    with open('./datasets/responses/init_' + str(args.index) + '_' + model_name + '.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the header only if the file didn't exist before
        if file.tell() == 0:
            writer.writerow(['question', 'Response'])

        num_q_per_worker = int(np.ceil(len(questions)/args.num_workers))
        start = args.index * num_q_per_worker
        end = min((args.index + 1) * num_q_per_worker, len(questions))
        for i in tqdm(range(start, end)):
            question = questions[i]
            for j in range(len(initial_seed)):
                print("Progress for this question is {} out of {}".format(j, len(initial_seed)))
                seed = initial_seed[j]
                mutant_replaced = replace_template(question, seed)
                while True:   #sometimes the query to OPENAI may fail, so we need to try again until success
                    try:
                        response = execute_query(args, mutant_replaced, MODEL, TOK)
                        writer.writerow([i, response])
                        break
                    except:
                        continue

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--index', type=int, default=10, help='task id')
    argparse.add_argument('--openai_key', type=str, default='You must have an OpenAI key', help='OpenAI key')
    argparse.add_argument('--model_path', type=str, default='gpt-3.5-turbo', help='openai model or open-sourced LLMs')
    argparse.add_argument('--num_workers', type=int, default=10, help='number of workers')
    argparse.add_argument("--temperature", type=float, default=0.01)                     # some models like ChatGLM do not support zero temperature
    argparse.add_argument("--repetition_penalty", type=float, default=1.0)
    argparse.add_argument("--max-new-tokens", type=int, default=512)
    argparse.add_argument("--debug", action="store_true")
    argparse.add_argument("--attack", type=bool, default=True, help="whether to attack the model with jailbreak prompts")
    add_model_args(argparse)
    args = argparse.parse_args()

    MODEL, TOK = prepare_model_and_tok(args)
    
    initial_seed = pd.read_excel('datasets/prompts/jailbreak-prompt.xlsx')['text'].tolist()
    questions = pd.read_csv('datasets/questions/question_list.csv')['text'].tolist()
    main(args, MODEL, TOK, initial_seed, questions) 