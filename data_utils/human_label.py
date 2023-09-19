import os
import pandas as pd
from termcolor import colored

base_path = '/home/jys3649/projects/jailbreak/datasets/responses_pre_filtered/'
save_path = '/home/jys3649/projects/jailbreak/datasets/responses_labeled/'
model_name = 'gpt-3.5-turbo'
max_files = 10


# Find the first file that hasn't been labeled yet
for file_num in range(max_files):
    filename = f'init_{file_num}_{model_name}_filtered.csv'
    save_filename = f'init_{file_num}_{model_name}_labeled.csv'
    full_filepath = os.path.join(base_path, filename)
    full_save_filepath = os.path.join(save_path, save_filename)

    # Check if the file exists and the labeled version doesn't
    if os.path.isfile(full_filepath) and not os.path.isfile(full_save_filepath):
        df = pd.read_csv(full_filepath)
        df['label'] = ''
        total_rows = len(df.index)

        idx = 0
        # Iterate over each row
        while idx < total_rows:
            row = df.iloc[idx]
            # Clear console output
            os.system('cls' if os.name == 'nt' else 'clear')

            # Calculate progress percentage
            progress = ((idx+1) / total_rows) * 100
            print(colored(f'Progress: {progress:.2f}%\n', 'cyan'))
            print(colored(f"Question: {row['question']}\nResponse: {row['Response']}", 'green'))

            if row['filter'] == 'reject':
                label = '0'
            elif row['filter'] == 'toxic':
                label = '1'
            else:
                # Wait for input
                if row['filter'] == 'high_toxic_score':
                    print('\n')
                    print(colored('High toxic score detected with moderation', 'red'))
                label = input(colored("Input label (press 'enter' for '0', input '1' for '1', type 'previous' to go back): ", 'blue'))
                
            # If 'previous' is entered, decrement the index and continue to next iteration
            if label.lower() == 'previous' and idx > 0:
                idx -= 1
                continue
            
            # Check if input is valid
            if label not in ['0', '1', '']:
                print(colored('Invalid input, please try again', 'red'))
                continue

            # Assign the label
            if label == '1':
                df.at[idx, 'Label'] = '1'
            else:
                df.at[idx, 'Label'] = '0'

            idx += 1

        # Delete the filter column
        df.drop(columns=['filter'], inplace=True)
        # Save the dataframe to a new csv file
        df.to_csv(full_save_filepath, index=False)
        
        # # If this is the last file, merge all labeled files
        # if file_num == max_files - 1:
        #     all_files = [pd.read_csv(os.path.join(save_path, f'init_{i}_{model_name}_labeled.csv')) for i in range(max_files)]
        #     merged_df = pd.concat(all_files, ignore_index=True)
        #     merged_df.to_csv(os.path.join(save_path, f'all_labeled_{model_name}.csv'), index=False)

        # After saving, break the loop
        break

