import torch
from torch import nn
import shap
import numpy as np
import pandas as pd
from joblib import load
from torch.utils.data import DataLoader, TensorDataset

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
    
class ModelWrapper:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, input_data):
        # Standardize input_data using the loaded scaler
        input_data_scaled = self.scaler.transform(input_data)
        
        # Convert to torch tensor
        input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1).numpy()
        return probabilities

def load_model_and_scaler():
    scaler = load('../neural_network/focus_scaler.joblib')
    input_size = 15
    num_classes = 2
    model = NeuralNetwork(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load('../neural_network/focus_best_model_checkpoint.pth'))
    model.eval()
    return model, scaler

def main():
    model, scaler = load_model_and_scaler()
    wrapped_model = ModelWrapper(model, scaler)
    
    df = pd.read_csv('../data_collection/train/Pz_AF3_AF4_combined_train_focus_data.csv')
    features = df[['POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma', 'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma', 'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma']]  # specify your feature columns here
    actual_data = features.to_numpy()

    # To select a random subset of 100 samples
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(actual_data.shape[0], 100, replace=False)
    subset_data = actual_data[indices, :]
    
    # Initialize SHAP explainer with a representative sample of your input data
    explainer = shap.KernelExplainer(wrapped_model.predict, subset_data)
    
    # Compute SHAP values for the actual data
    shap_values = explainer.shap_values(subset_data, nsamples='auto')  # Adjust nsamples as needed

    print(shap_values)
    shap_values_positive_class = shap_values[:, :, 1]
    feature_names=['POW.AF3.Theta', 'POW.AF3.Alpha', 'POW.AF3.BetaL', 'POW.AF3.BetaH', 'POW.AF3.Gamma', 'POW.AF4.Theta', 'POW.AF4.Alpha', 'POW.AF4.BetaL', 'POW.AF4.BetaH', 'POW.AF4.Gamma', 'POW.Pz.Theta', 'POW.Pz.Alpha', 'POW.Pz.BetaL', 'POW.Pz.BetaH', 'POW.Pz.Gamma']
    
    # Generate a summary plot for the overall feature importance
    shap.summary_plot(shap_values_positive_class, subset_data, feature_names=feature_names)
    shap.summary_plot(shap_values_positive_class, subset_data, plot_type="bar", feature_names=feature_names)
    
if __name__ == '__main__':
    main()
