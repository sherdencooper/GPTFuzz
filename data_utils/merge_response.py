import pandas as pd
import os
import glob

def merge_csv_files(model_name, folder_path):
    # skip if the merged csv file already exists
    if os.path.exists(os.path.join(folder_path, f"init_{model_name}_merged.csv")):
        return
    
    # sort the csv files based on their order
    csv_files = sorted(glob.glob(os.path.join(folder_path, f"init_*_{model_name}.csv")))
    
    # create an empty list to hold dataframes
    df_list = []
    
    # loop over the list of csv files
    for csv in csv_files:
        # read the csv file and append it to the list
        df = pd.read_csv(csv)
        df_list.append(df)

    # concatenate all the dataframes in the list
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # write the concatenated dataframe to new csv file
    merged_df.to_csv(os.path.join(folder_path, f"init_{model_name}_merged.csv"), index=False)

# Example usage:
merge_csv_files('gpt-3.5-turbo', '/home/jys3649/projects/jailbreak/datasets/responses')
merge_csv_files('vicuna-7b-v1.3', '/home/jys3649/projects/jailbreak/datasets/responses')
merge_csv_files('llama-2-7b-chat-hf', '/home/jys3649/projects/jailbreak/datasets/responses')
