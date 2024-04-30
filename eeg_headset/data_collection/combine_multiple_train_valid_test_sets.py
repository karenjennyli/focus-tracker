import pandas as pd
import glob
import os
from sklearn.preprocessing import StandardScaler

# Define the path to the folder containing the CSV files
folder_path = 'valid'  # Update this to your folder path

# Use glob to get all the CSV files in the folder
# csv_files = glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_*rohan*_focus.csv"))
# append to csv_files the csv files ending with distracted.csv
# csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_*rohan*_distracted.csv"))
# csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_*arnav*_focus.csv"))
# csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_*arnav*_distracted.csv"))

# csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_" + folder_path + "_labeled_karen_take_1_focus.csv"))
# csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_" + folder_path + "_labeled_karen_take_2_distracted.csv"))
# csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_" + folder_path + "_karen_take_3_focus.csv"))
# csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_" + folder_path + "_karen_take_4_distracted.csv"))

csv_files = glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_*justin*.csv"))
csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_*ishan*.csv"))
csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_*rick_*.csv"))

# csv_files = glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_filtered_karen_take_5_distracted.csv"))
# csv_files += glob.glob(os.path.join(folder_path, "Pz_AF3_AF4_filtered_karen_take_6_focus.csv"))

print(csv_files)

# List to hold data from each CSV file
dataframes_list = []

# Input feature columns to be normalized
input_columns = ['POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma',
                 'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma',
                 'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma']

# Iterate over the list of CSV files
for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Normalize the input feature columns using Z-score normalization
    # scaler = StandardScaler()
    # df[input_columns] = scaler.fit_transform(df[input_columns])
    
    # Append the normalized dataframe to the list
    dataframes_list.append(df)

# Concatenate all dataframes in the list into one
combined_df = pd.concat(dataframes_list, ignore_index=True)

# Save the combined and normalized dataframe to a new CSV file
combined_csv_path = os.path.join(folder_path, 'Pz_AF3_AF4_combined_' + folder_path + '_flow_data.csv')
combined_df.to_csv(combined_csv_path, index=False)

print(f"Combined and normalized CSV created at: {combined_csv_path}")
