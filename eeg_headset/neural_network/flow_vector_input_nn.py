from joblib import dump, load
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Neural Network definition
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 32)
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
    label_mapping = {-1.0: 0, 0.0: 0, 1.0: 1}
    labels = df[label_column].map(label_mapping).values
    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)
    # dump(scaler, 'flow_scaler.joblib')
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

# Dataset class
class EEGDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# Load datasets
input_columns = ['POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma', 'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma','POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma']
label_column = 'label'
train_inputs, train_labels = load_dataset('../data_collection/train/Pz_AF3_AF4_combined_train_flow_data.csv', input_columns, label_column)
val_inputs, val_labels = load_dataset('../data_collection/valid/Pz_AF3_AF4_combined_valid_flow_data.csv', input_columns, label_column)

# Create dataloaders
train_dataset = EEGDataset(train_inputs, train_labels)
val_dataset = EEGDataset(val_inputs, val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# Initialize the Neural Network, Loss Function, and Optimizer
model = NeuralNetwork(input_size=15, num_classes=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training and Validation
epochs = 100
best_val_loss = float('inf')
train_losses = []  # List to store average training losses per epoch
val_losses = []  # List to store average validation losses per epoch

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)  # Append average training loss
    print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}')
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)  # Append average validation loss
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}')
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'flow_best_model_checkpoint.pth')
        print('Best model saved!')

# Plotting the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
