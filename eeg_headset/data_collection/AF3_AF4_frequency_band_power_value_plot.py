import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV data into a pandas DataFrame
df = pd.read_csv('train/AF3_AF4_train_labeled_ishan_take_1_raw_data.csv')

df = df[df['label'] != 0.0]

# Create a color column based on the label
df['color'] = df['label'].map({-1.0: 'red', 1.0: 'green', 0.0: 'blue'})

# Set up the subplots in a 2x2 grid
fig, axs = plt.subplots(3, 2, figsize=(10, 10))  # Adjust the size as needed

# Scatter plot for EEG.AF3
axs[0, 0].scatter(df['Timestamp'], df['EEG.AF3'], c=df['color'], label='EEG.AF3')
axs[0, 0].set_title('EEG.AF3')
axs[0, 0].set_ylabel('Power Values')

# Scatter plot for EEG.AF4
axs[0, 1].scatter(df['Timestamp'], df['EEG.AF4'], c=df['color'], label='EEG.AF4')
axs[0, 1].set_title('EEG.AF4')

# Scatter plot for POW.AF3.Theta
axs[1, 0].scatter(df['Timestamp'], df['POW.AF3.Theta'], c=df['color'], label='POW.AF3.Theta')
axs[1, 0].set_title('POW.AF3.Theta')
axs[1, 0].set_ylabel('Power Values')

# Scatter plot for POW.AF3.Alpha
axs[1, 1].scatter(df['Timestamp'], df['POW.AF3.Alpha'], c=df['color'], label='POW.AF3.Alpha')
axs[1, 1].set_title('POW.AF3.Alpha')

# Scatter plot for POW.AF4.Theta
axs[2, 0].scatter(df['Timestamp'], df['POW.AF4.Theta'], c=df['color'], label='POW.AF4.Theta')
axs[2, 0].set_title('POW.AF4.Theta')
axs[2, 0].set_xlabel('Timestamp')
axs[2, 0].set_ylabel('Power Values')

# Scatter plot for POW.AF4.Alpha
axs[2, 1].scatter(df['Timestamp'], df['POW.AF4.Alpha'], c=df['color'], label='POW.AF4.Alpha')
axs[2, 1].set_title('POW.AF4.Alpha')
axs[2, 1].set_xlabel('Timestamp')

# Add legends and adjust layout
for ax in axs.flat:
    ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
    ax.legend()

plt.tight_layout()
plt.show()

'''
# Scatter plot for POW.Pz.Theta
axs[1, 0].scatter(df['Timestamp'], df['POW.Pz.Theta'], c=df['color'], label='POW.Pz.Theta')
axs[1, 0].set_title('POW.Pz.Theta')
axs[1, 0].set_ylabel('Power Values')

# Scatter plot for POW.Pz.Alpha
axs[1, 1].scatter(df['Timestamp'], df['POW.Pz.Alpha'], c=df['color'], label='POW.Pz.Alpha')
axs[1, 1].set_title('POW.Pz.Alpha')
'''

