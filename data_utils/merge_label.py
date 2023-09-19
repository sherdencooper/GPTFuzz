import pandas as pd

# Create a list of filenames
filenames = [f"init_{i}_gpt-3.5-turbo_labeled.csv" for i in range(8,10)]
# filenames = [f"initial_temp_labeled/{i}_initial_seed_labeled.csv" for i in range(10)]

# Create a list to hold the dataframes
df_list = []

# Loop through the files and read them in with pandas
for filename in filenames:
    data = pd.read_csv(filename)
    df_list.append(data)

# Concatenate all the dataframes together
combined_df = pd.concat(df_list, ignore_index=True)

# Write the combined dataframe to a new CSV file
combined_df.to_csv("evaluate.csv", index=False)