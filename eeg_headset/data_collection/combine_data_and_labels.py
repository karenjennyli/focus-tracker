import pandas as pd

# Load the CSV files
raw_data_path = 'recordings/ricky_take_1_raw_data.csv'
labels_path = 'labels/ricky take 1 focus_data_2024-04-19 13:19:42.489.csv'

raw_data_df = pd.read_csv(raw_data_path)
labels_df = pd.read_csv(labels_path)

# Initialize a new column in raw_data_df for labels with a default value (e.g., NaN or a placeholder for 'unlabeled')
raw_data_df['label'] = None

# Map from your textual labels to numeric ones, if necessary
label_mapping = {'Focused': 1, 'Neutral': 0, 'Distracted': -1}

# Find the timestamp of the last label entry
last_label_timestamp = labels_df['timestamp_epoch'].max()

# Iterate through the labels dataframe and assign labels to raw data based on the timestamp ranges
for _, label_row in labels_df.iterrows():
    label_timestamp = label_row['timestamp_epoch']
    label_value = label_row['focus_state']  # Assuming 'focus_state' is the column with Focused/Neutral/Distracted
    numeric_label = label_mapping[label_value]
    
    # Apply the current label to the raw data rows where applicable
    raw_data_df.loc[(raw_data_df['Timestamp'] >= label_timestamp) & (raw_data_df['Timestamp'] < last_label_timestamp), 'label'] = numeric_label

# Assign None to all entries with a timestamp greater than or equal to the last label timestamp
raw_data_df.loc[raw_data_df['Timestamp'] >= last_label_timestamp, 'label'] = None

# Save the modified dataframe back to a CSV
modified_raw_data_path = 'labeled_data/' + 'labeled_' + raw_data_path.split('/')[-1]
columns_to_export = ['Timestamp', 'EEG.AF3', 'EEG.Pz', 'EEG.AF4', 'CQ.AF3', 'CQ.Pz', 'CQ.AF4','CQ.Overall', 'EQ.OVERALL', 'EQ.AF3', 'EQ.Pz', 'EQ.AF4', 'PM.Attention.IsActive', 'PM.Attention.Scaled', 'PM.Attention.Raw', 'PM.Attention.Min', 'PM.Attention.Max', 'PM.Engagement.IsActive', 'PM.Engagement.Scaled', 'PM.Engagement.Raw', 'PM.Engagement.Min', 'PM.Engagement.Max', 'PM.Excitement.IsActive', 'PM.Excitement.Scaled', 'PM.Excitement.Raw', 'PM.Excitement.Min', 'PM.Excitement.Max', 'PM.LongTermExcitement', 'PM.Stress.IsActive', 'PM.Stress.Scaled', 'PM.Stress.Raw', 'PM.Stress.Min', 'PM.Stress.Max', 'PM.Relaxation.IsActive', 'PM.Relaxation.Scaled', 'PM.Relaxation.Raw', 'PM.Relaxation.Min', 'PM.Relaxation.Max', 'PM.Interest.IsActive', 'PM.Interest.Scaled', 'PM.Interest.Raw', 'PM.Interest.Min', 'PM.Interest.Max', 'PM.Focus.IsActive', 'PM.Focus.Scaled', 'PM.Focus.Raw', 'PM.Focus.Min', 'PM.Focus.Max', 'POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma', 'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma', 'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma', 'label']
raw_data_df.to_csv(modified_raw_data_path, columns=columns_to_export, index=False)

print(f'Modified raw data saved to {modified_raw_data_path}')
