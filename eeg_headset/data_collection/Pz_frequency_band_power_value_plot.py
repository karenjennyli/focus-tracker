import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV data into a pandas DataFrame
df = pd.read_csv('train/Pz_train_labeled_ishan_take_1_raw_data.csv')

df = df[df['label'] != 0.0]

# Create a color column based on the label
df['color'] = df['label'].map({-1.0: 'red', 1.0: 'green', 0.0: 'blue'})

# Set up the subplots in a 3x1 grid
fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # Adjust the size as needed

# Scatter plot for EEG.Pz
axs[0].scatter(df['Timestamp'], df['EEG.Pz'], c=df['color'], label='EEG.Pz')
axs[0].set_title('EEG.Pz')
axs[0].set_ylabel('Power Values')

# Scatter plot for EEG.AF4
# axs[0, 1].scatter(df['Timestamp'], df['EEG.AF4'], c=df['color'], label='EEG.AF4')
# axs[0, 1].set_title('EEG.AF4')

# Scatter plot for POW.Pz.Theta
axs[1].scatter(df['Timestamp'], df['POW.Pz.Theta'], c=df['color'], label='POW.Pz.Theta')
axs[1].set_title('POW.Pz.Theta')
axs[1].set_ylabel('Power Values')

# Scatter plot for POW.Pz.Alpha
axs[2].scatter(df['Timestamp'], df['POW.Pz.Alpha'], c=df['color'], label='POW.Pz.Alpha')
axs[2].set_title('POW.Pz.Alpha')
axs[2].set_ylabel('Power Values')
axs[2].set_xlabel('Timestamp')


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

