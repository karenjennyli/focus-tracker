import pandas as pd
import glob
import os

# Define the path to the folder containing the CSV files
folder_path = 'valid'  # Update this to your folder path

# Use glob to get all the CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "Pz_*.csv"))

# List to hold data from each CSV file
dataframes_list = []

# Iterate over the list of CSV files
for csv_file in csv_files:
    # Read the CSV file and append it to the list
    df = pd.read_csv(csv_file)
    dataframes_list.append(df)

# Concatenate all dataframes in the list into one
combined_df = pd.concat(dataframes_list, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_csv_path = folder_path + '/' + 'Pz_combined_' + folder_path + '_data.csv'
combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined CSV created at: {combined_csv_path}")
