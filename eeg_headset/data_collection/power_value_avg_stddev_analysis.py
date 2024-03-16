import pandas as pd

# File path to the CSV
csv_file_path = 'labeled_data/labeled_ishan_take_1_raw_data.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# AF3 and AF4 power values for all frequency bands
# columns = [
#     'POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma',
#     'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma'
# ]

# AF3 and AF4 power values for focus related frequency bands
# columns = [
#     'POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF4.Theta', 'POW.AF4.Alpha'
# ]

# Pz power values for all frequency bands
# columns = [
#     'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma'
# ]

# Pz power values for focus related frequency bands
# columns = [
#     'POW.Pz.Theta', 'POW.Pz.Alpha'
# ]

# PM scaled values
columns = [
    'PM.Attention.Scaled', 'PM.Engagement.Scaled', 'PM.Excitement.Scaled', 'PM.LongTermExcitement', 'PM.Stress.Scaled', 'PM.Relaxation.Scaled', 'PM.Interest.Scaled', 'PM.Focus.Scaled'
]

# Calculate the average values for each label (0.0, -1.0, and 1.0)
average_values = df.groupby('label')[columns].mean()

# Calculate the standard deviations for each label (0.0, -1.0, and 1.0)
standard_deviations = df.groupby('label')[columns].std()

# Print the average values and standard deviations in an easy-to-read format
print("Average Values:\n", average_values)
print("\nStandard Deviations:\n", standard_deviations)


'''
# Calculate the average values for each label
average_focused = df[df['label'] == 1.0][columns_to_average].mean()
average_neutral = df[df['label'] == 0.0][columns_to_average].mean()
average_distracted = df[df['label'] == -1.0][columns_to_average].mean()

# Print the results
print("Averages for label 1.0 (Focused):")
print(average_focused)
print("\nAverages for label 0.0 (Neutral):")
print(average_neutral)
print("\nAverages for label -1.0 (Distracted):")
print(average_distracted)
'''