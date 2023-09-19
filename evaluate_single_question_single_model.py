import pandas as pd
from tqdm import tqdm
import csv
import os
import numpy as np
import argparse
import fnmatch

def main(args):
    model_name = args.model_path.split('/')[-1]
    directory_path = '/home/jys3649/projects/jailbreak/datasets/prompts_generated/single_single'
    if args.initial_seed_filter:
        matching_files = fnmatch.filter(os.listdir(directory_path), f'single_single_{model_name}_{args.filter_method}_*.csv')
    else:
        matching_files = fnmatch.filter(os.listdir(directory_path), f'single_single_{model_name}_*.csv')
    full_paths = [os.path.join(directory_path, filename) for filename in matching_files]
    success_num = 0
    query_total = 0
    unique_parent_indices = set()
    for path in full_paths:
        df = pd.read_csv(path)
        text = df['text'].tolist()
        query = df['num_query'].tolist()
        if len(text) == 0:
            query_total += 500
        else:
            success_num += 1
            query_total += query[-1]
            unique_parent_indices.update(df['parent_index'].unique())
    print(f'For model {model_name}, the success rate is {success_num/len(full_paths)}, the average query is {query_total/len(full_paths)}')
    print(f'For model {model_name}, the number of unique parent indices is {len(unique_parent_indices)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='openai model or open-sourced LLMs')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--initial_seed_filter", type=bool, default=True) 
    parser.add_argument("--filter_method", type=str, default='0-only')
    args = parser.parse_args()
    main(args)