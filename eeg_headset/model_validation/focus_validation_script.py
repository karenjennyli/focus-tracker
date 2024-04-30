import pandas as pd
import torch
import torch.nn as nn
from joblib import load
import numpy as np

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

def process_data(csv_path):
    # Load the CSV file
    data = pd.read_csv(csv_path)

    print(data.columns)
    
    # Filter the data
    # filtered_data = data[(data['EQ.AF3'] == 4.0) & (data['EQ.AF4'] == 4.0) & (data['EQ.Pz'] == 4.0)]
    filtered_data = data
    
    scaler = load('../neural_network/focus_scaler.joblib')
    input_size = 15
    num_classes = 2
    model = NeuralNetwork(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load('../neural_network/focus_best_model_checkpoint.pth'))
    model.eval()
    
    focusedCount = 0
    distractedCount = 0
    
    for index, row in filtered_data.iterrows():
        input_features = row[['POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma', 
                              'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma',
                              'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma']].values
        
        # Check if any of the selected features are missing in this row
        if np.isnan(input_features).any():
            continue  # Skip this row and move to the next one
                              
        input_vector_scaled = scaler.transform([input_features])
        input_tensor = torch.tensor(input_vector_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        class_names = ['Distracted', 'Focused']
        predicted_class = class_names[predicted.item()]

        if predicted_class == 'Focused':
            focusedCount += 1
        else:
            distractedCount += 1
    
    print(f"Total Focused: {focusedCount}")
    print(f"Total Distracted: {distractedCount}")

if __name__ == "__main__":
    csv_path = '../data_collection/filtered_data/Pz_AF3_AF4_filtered_karen_take_3_focus.csv'  # Update this to the path of your CSV file
    process_data(csv_path)
