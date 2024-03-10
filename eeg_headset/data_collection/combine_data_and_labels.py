import pandas as pd

# Load the CSV files
raw_data_path = 'recordings/ishan_take_1_raw_data.csv'
labels_path = 'labels/ishan take 1 focus_data_2024-02-26 11:02:18.766.csv'

raw_data_df = pd.read_csv(raw_data_path)
labels_df = pd.read_csv(labels_path)

# Initialize a new column in raw_data_df for labels with a default value (e.g., NaN or a placeholder for 'unlabeled')
raw_data_df['label'] = None

# Map from your textual labels to numeric ones, if necessary
label_mapping = {'Focused': 1, 'Neutral': 0, 'Distracted': -1}

# Iterate through the labels dataframe and assign labels to raw data based on the timestamp ranges
current_label = None
for _, label_row in labels_df.iterrows():
    label_timestamp = label_row['timestamp_epoch']
    label_value = label_row['focus_state']  # Assuming 'label' is the column with Focused/Neutral/Distracted
    numeric_label = label_mapping[label_value]
    
    # Apply the current label to the raw data rows where applicable
    raw_data_df.loc[(raw_data_df['Timestamp'] >= label_timestamp), 'label'] = numeric_label
    
    current_label = numeric_label

# Save the modified dataframe back to a CSV
modified_raw_data_path = 'labeled_data/' + 'modified_' + raw_data_path.split('/')[-1]
columns_to_export = ['Timestamp', 'EEG.AF3', 'EEG.AF4', 'CQ.AF3', 'CQ.AF4', 'CQ.Overall', 'EQ.OVERALL', 'EQ.AF3', 'EQ.AF4', 'POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma', 'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma', 'label']
raw_data_df.to_csv(modified_raw_data_path, columns=columns_to_export, index=False)

print(f'Modified raw data saved to {modified_raw_data_path}')
