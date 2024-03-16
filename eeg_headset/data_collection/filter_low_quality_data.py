import pandas as pd

# Load the CSV files
labeled_data_path = 'labeled_data/labeled_justin_take_5_raw_data.csv'

labeled_df = pd.read_csv(labeled_data_path)

# Remove data entries without a label
labeled_df.dropna(subset=['label'], inplace=True)

# Remove data entries without frequency band power values
labeled_df.dropna(subset=['POW.AF3.Theta'], inplace=True)

# Find indices where both EQ.AF3 and EQ.AF4 have the value 4
# eq_indices = labeled_df[(labeled_df['EQ.AF3'] == 4.0) & (labeled_df['EQ.AF4'] == 4.0)].index

# Find indices where both EQ.Pz has the value 4
eq_indices = labeled_df[(labeled_df['EQ.Pz'] == 4.0)].index

# Find consecutive pairs of indices where EQ values are 4 for both columns
pairs_of_interest = [(eq_indices[i], eq_indices[i + 1]) for i in range(len(eq_indices) - 1) if eq_indices[i + 1] - eq_indices[i] == 64]

# Initialize an empty list to store DataFrames that you want to concatenate later
dfs_to_concatenate = []

# Loop through pairs and add the data range to the list
for start, end in pairs_of_interest:
    dfs_to_concatenate.append(labeled_df.loc[start:end])

# Concatenate all DataFrames in the list
filtered_df = pd.concat(dfs_to_concatenate)

# Reset index
filtered_df = filtered_df.reset_index(drop=True)

# split filtered data into train (80%), validation(10%), and test(10%) sets
train = filtered_df.sample(frac=0.8,random_state=200)
valid_test = filtered_df.drop(train.index)
valid = valid_test.sample(frac=0.5,random_state=200)
test = valid_test.drop(valid.index)

# reset indices
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

# Specify your file name
file_name = 'filtered_data/' + 'Pz_filtered_' + labeled_data_path.split('/')[-1]

# Export the DataFrame to a CSV file
filtered_df.to_csv(file_name, index=False)

print(f'Data exported to {file_name}')

# Specify your train, valid, test file names
train_file_name = 'train/' + 'Pz_train_' + labeled_data_path.split('/')[-1]
valid_file_name = 'valid/' + 'Pz_valid_' + labeled_data_path.split('/')[-1]
test_file_name = 'test/' + 'Pz_test_' + labeled_data_path.split('/')[-1]

# Export the DataFrame to a CSV file
train.to_csv(train_file_name, index=False)
valid.to_csv(valid_file_name, index=False)
test.to_csv(test_file_name, index=False)

print(f'Data exported to {train_file_name}')
print(f'Data exported to {valid_file_name}')
print(f'Data exported to {test_file_name}')
