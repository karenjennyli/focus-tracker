import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming the NeuralNetwork class and load_dataset function are defined as previously mentioned# Neural Network definition
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)  # Output layer, no activation here
        return x

# Function to load dataset
def load_dataset(csv_file, input_columns, label_column):
    df = pd.read_csv(csv_file)
    inputs = df[input_columns].values
    
    # Map labels from -1.0, 0.0, 1.0 to 0 for not in flow and 1 for flow
    label_mapping = {-1.0: 0, 0.0: 0, 1.0: 1}
    labels = df[label_column].map(label_mapping).values
    
    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)
    
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are long type for CrossEntropyLoss
    
    return inputs, labels

# Load the test dataset
input_columns = ['EEG.AF3', 'POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma', 'EEG.AF4', 'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma', 'EEG.Pz', 'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma']
label_column = 'label'
test_inputs, test_labels = load_dataset('../data_collection/test/Pz_AF3_AF4_combined_test_data.csv', input_columns, label_column)

# Create a Dataset for the test data
class EEGTestDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

test_dataset = EEGTestDataset(test_inputs, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Load the model with the best validation loss
model = NeuralNetwork(input_size=18, num_classes=2)
model.load_state_dict(torch.load('best_model_checkpoint.pth'))
model.eval()  # Set the model to evaluation mode

# Predictions and actual labels
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Optionally, print or plot the confusion matrix
print(cm)

# Calculate precision, recall, and F1 score
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')

class_names = ['Not in Flow', 'Flow']

# Plotting using matplotlib and seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
