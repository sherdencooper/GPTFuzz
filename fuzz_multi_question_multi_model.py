import openai
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
import subprocess
import numpy as np
from fuzz_utils import *
from llm_utils.creat_model import prepare_model_and_tok
import random
import argparse
import copy
from fastchat.model import add_model_args

def main(args, args_target, MODEL, TOK, MODEL_TARGET, TOK_TARGET, questions, initial_seed):
    target_model_name = 'chatgpt_vicuna_llama2'

    if args.initial_seed_filter:
        if args.filter_method == 'filter-failed':
            failed_seed_index = np.load(f'./datasets/responses/{target_model_name}_failed_seeds.npy')
            #delete the failed seed from initial seed pool
            initial_seed = np.delete(initial_seed, failed_seed_index, axis=0)
        elif args.filter_method == '0-only':
            failed_seed_index = np.load(f'./datasets/responses/{target_model_name}_failed_seeds.npy')
            initial_seed = [initial_seed[i] for i in failed_seed_index]
        elif args.filter_method == 'top-1':
            top_1_index = np.load(f'./datasets/responses/{target_model_name}_top_1_seed_index.npy')
            initial_seed = initial_seed[top_1_index]
        elif args.filter_method == 'top-5':
            top_5_index = np.load(f'./datasets/responses/{target_model_name}_top_5_seed_index.npy')
            initial_seed = [initial_seed[i] for i in top_5_index]
        elif args.filter_method == 'bottom-20':
            bottom_30_seed_index = np.load(f'./datasets/responses/{target_model_name}_bottom_20_seed_index.npy')
            initial_seed = [initial_seed[i] for i in bottom_30_seed_index]
        else:
            raise NotImplementedError

    status = fuzzing_status(questions, initial_seed=initial_seed, max_query=args.max_query, energy=args.energy, seed_selection_strategy=args.seed_selection_strategy, mode='multi-multi')
    counter = 0
    save_len = len(status.seed_queue)
    if args.initial_seed_filter:
        save_path = './datasets/prompts_generated/multi_multi/multi_multi_' + target_model_name + '_' + args.filter_method + '.csv'
    else:
        save_path = './datasets/prompts_generated/multi_multi/multi_multi_' + target_model_name + '.csv'
    # delete the file if it exists
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'w', newline='') as file:
        print("The multi-model multi-question fuzzing process is started!")
        print("The file is saved as {}".format(save_path))
        writer = csv.writer(file)
        writer.writerow(['text', 'parent_index', 'generation', 'mutation', 'index', 'children_index', 'response', 'visited_num', 'num_query'])
        while True:
            selected_seed = status.seed_selection_strategy()
            mutate_results, mutation = status.mutate_strategy(selected_seed, status, MODEL, TOK, args)
            # print(mutate_results['choices'][0]['message']['content'])
            # print(mutate_results[-1])
            attack_results, valid_input_index, data = execute(status, mutate_results, args_target, MODEL_TARGET, TOK_TARGET)
            status.update(attack_results, mutate_results, mutation, valid_input_index, data)
            counter += 1
            if counter % 1 == 0:
                print_status(status)

            if len(status.seed_queue) > save_len:
                for i in range(save_len, len(status.seed_queue)):
                    if status.seed_queue[i].parent != 'root':
                        writer.writerow([status.seed_queue[i].text, status.seed_queue[i].parent_index, status.seed_queue[i].generation, status.seed_queue[i].mutation, status.seed_queue[i].index, status.seed_queue[i].children_index, status.seed_queue[i].response, status.seed_queue[i].visited_num, status.query])
                save_len = len(status.seed_queue)

            if status.stop_condition():
                break

    print("Fuzzing finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--openai_key', type=str, default='You must have an OpenAI key', help='OpenAI key')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo', help='openai model or open-sourced LLMs')
    parser.add_argument('--target_model', type=list, default=['meta-llama/Llama-2-7b-chat-hf', 'gpt-3.5-turbo', 'lmsys/vicuna-7b-v1.3'], help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=30000, help='The maximum number of queries')
    parser.add_argument('--energy', type=int, default=1, help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str, default='MCTS', help='The seed selection strategy')
    parser.add_argument("--temperature", type=float, default=1.0)          
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--initial_seed_filter", type=bool, default=False)  # seed initialization
    parser.add_argument("--filter_method", type=str, default='top-5')
    add_model_args(parser)
    args = parser.parse_args()
    assert args.energy == 1, "The energy for multi-question fuzzing now only supports 1!"
    args.num_gpus = 3
    MODEL, TOK = prepare_model_and_tok(args)
    args_target = copy.deepcopy(args)
    args_target.model_path = args.target_model
    args_target.temperature = 0.01   #some models need to have strict positive temperature
    MODEL_TARGET, TOK_TARGET = prepare_model_and_tok(args_target)
    questions = pd.read_csv('datasets/questions/question_list.csv')['text'].tolist()[:20]   #use 20 questions to optimize the fuzzing process
    initial_seed = pd.read_excel('./datasets/prompts/jailbreak-prompt.xlsx')['text'].tolist()
    print(args)
    main(args, args_target, MODEL, TOK, MODEL_TARGET, TOK_TARGET, questions, initial_seed)