import numpy as np
import os
import pandas as pd
# from llm_utils.roberta_utils import predict   #comment this line to accelerate the analysis
from tqdm import tqdm

def initial_response_analysis(batch_size=640):
    # model_names = ['vicuna-7b-v1.3', 'gpt-3.5-turbo', 'Llama-2-7b-chat-hf', 'gpt-3.5-turbo-0301']
    model_names = ['vicuna-7b-v1.3', 'gpt-3.5-turbo', 'Llama-2-7b-chat-hf']
    for name in model_names:
        file_name = f'./datasets/responses/init_{name}_merged'
        # skip if the txt file already exists
        if os.path.exists(file_name + '_labels.txt'):
            continue
        df = pd.read_csv(file_name + '.csv')
        responses = df['Response'].tolist()
        # Predict labels for all responses
        predicted_labels = []
        for i in tqdm(range(0, len(responses), batch_size)):
            batch = responses[i:i+batch_size]
            batch_labels = predict(batch)
            batch_labels = [label.item() for label in batch_labels]
            predicted_labels.extend(batch_labels)
        
        # Save predicted labels to a txt file
        with open(file_name + "_labels.txt", 'w') as f:
            for label in predicted_labels:
                f.write(str(label) + '\n')

    predict_results = np.zeros((len(model_names), 7700)) # 77 * 100 = 7700 responses
    for name in model_names:
        file_name = f'./datasets/responses/init_{name}_merged'
        with open(file_name + '_labels.txt', 'r') as f:
            labels = f.readlines()
        labels = [int(label.strip()) for label in labels]
        predict_results[model_names.index(name)] = labels
    
    initial_seed_attack_result = np.zeros((77, 100, len(model_names)))
    for i in range(77):
        for j in range(100):
            initial_seed_attack_result[i][j] = predict_results[:, j * 77 + i]
    

    for idx, name in enumerate(model_names):
        single_model_result = initial_seed_attack_result[:, :, idx]
        # if any seed is successful, the attack is successful
        single_model_single_question_results = np.sum(single_model_result, axis=0) / 77
        for i in range(100):
            if single_model_single_question_results[i] > 0:
                single_model_single_question_results[i] = 1
        print(f'{name} single-question single-model ASR: {np.sum(single_model_single_question_results) / 100}')
        print(f'{name} single-question single-model average jailbreak number: {np.sum(np.sum(single_model_result, axis=0))/100}')
        # print the question index that the attack is not successful
        failed_question_index = np.where(single_model_single_question_results == 0)[0]
        if len(failed_question_index) > 0:
            print(f'{name} single-model single-question failed question index: {failed_question_index}')

        # top-1 result
        single_model_multi_question_results = np.sum(single_model_result, axis=1) / 100
        print(single_model_multi_question_results)
        print(f'{name} single-model multi-question top-1 highest ASR: {np.max(single_model_multi_question_results)}')
        # top-5 result
        top_five_result = top_n_result(single_model_multi_question_results, single_model_result, 5)
        print(f'{name} single-model multi-question top-5 ASR: {np.sum(top_five_result) / 100}')
        # print how many seeds succeed in zero question and save those seeds
        failed_seeds_index = np.where(single_model_multi_question_results == 0)[0]
        if len(failed_seeds_index) > 0:
            print(f'{name} single-model multi-question failed seeds: {failed_seeds_index}')
            np.save(f'./datasets/responses/{name}_failed_seeds.npy', failed_seeds_index)

        #save the index of the seed with top-1 ASR and top-5 ASR
        top_1_seed_index = np.argmax(single_model_multi_question_results)
        top_5_seed_index = np.argsort(single_model_multi_question_results)[::-1][:5]
        np.save(f'./datasets/responses/{name}_top_1_seed_index.npy', top_1_seed_index)
        np.save(f'./datasets/responses/{name}_top_5_seed_index.npy', top_5_seed_index)
        #save the index of the bottom 30 seed with lowest ASR
        bottom_20_seed_index = np.argsort(single_model_multi_question_results)[:20]
        np.save(f'./datasets/responses/{name}_bottom_20_seed_index.npy', bottom_20_seed_index)


        #evaluate those saved seed index on both train set and test set
        single_model_result_train = single_model_result[:, :20]
        single_model_result_test = single_model_result[:, 20:]
        train_and_test_response_analysis(name, 'bottom-20', single_model_result_train, single_model_result_test, bottom_20_seed_index[-1], bottom_20_seed_index[-5:][::-1])
        train_and_test_response_analysis(name, 'all', single_model_result_train, single_model_result_test, top_1_seed_index, top_5_seed_index)

    # select best 5 seed across all models
    best_5_seed_index = np.argsort(np.sum(np.sum(initial_seed_attack_result, axis=2), axis=1))[::-1][:5]
    print(f'best 5 seed index: {best_5_seed_index}')


def train_and_test_response_analysis(name, filter_name, single_model_result_train, single_model_result_test, top_1_index, top_5_index):
    train_top_one = top_n_test(single_model_result_train, top_1_index, question_num=20)
    train_top_five = top_n_test(single_model_result_train, top_5_index, question_num=20)
    test_top_one = top_n_test(single_model_result_test, top_1_index, question_num=80)
    test_top_five = top_n_test(single_model_result_test, top_5_index, question_num=80)
    print(f'{name} single-model multi-question top-one {filter_name} train ASR: {train_top_one}')
    print(f'{name} single-model multi-question top-five {filter_name} train ASR: {train_top_five}')
    print(f'{name} single-model multi-question top-one {filter_name} test ASR: {test_top_one}')
    print(f'{name} single-model multi-question top-five {filter_name} test ASR: {test_top_five}')


def top_n_test(single_model_result, test_seed_index, question_num=100):
    success_num = 0
    if test_seed_index.size == 1:
        test_seed_index = np.array([test_seed_index])
    for i in range(question_num):
        for test_seed in test_seed_index:
            if single_model_result[test_seed][i] == 1:
                success_num += 1
                break
    return success_num / question_num


def top_n_result(single_model_multi_question_results, single_model_result, n, question_num=100):
    seed_rank = np.argsort(single_model_multi_question_results)[::-1]
    top_n_results = np.zeros(question_num)
    for i in range(question_num):
        #it get success if the attack is successful in any of the n seeds
        top_n_results[i] = 1 if np.sum(single_model_result[seed_rank[:n], i]) > 0 else 0
    return top_n_results

initial_response_analysis()
