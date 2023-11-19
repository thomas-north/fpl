import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from simple_nn import NeuralNetwork

def load_data(season, gameweek):
    dataset = f'../data/{season}/gws/gw{gameweek}.csv'
    data = pd.read_csv(dataset, encoding="ISO-8859-1")
    # Feature engineering and scaling can be done here
    return data

def predict(model, input_data):
    with torch.no_grad():
        predictions = model(input_data)
    return predictions.numpy()

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model

# Hyperparameter configuration
config = {
    'input_size': 5,  # Adjust based on your actual input size
    'num_layers': 2,
    'hidden_size': 128,
    'output_size': 1,
    'learning_rate': 0.001,
    'batch_size': 128,
    'num_epochs': 200
}

# Load testing data
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# Instantiate the model for testing
model = NeuralNetwork(**config)

# Load the trained model
model = load_model(model, 'trained_model.pth')

# Make predictions on the test set
predictions = predict(model, X_test_tensor)

# Convert predictions back to a numpy array
y_pred_nn = predictions

# Load true labels for comparison
y_test = load_data(2022, 1)['total_points'].values

# Evaluate the model
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network Mean Squared Error: {mse_nn}')
