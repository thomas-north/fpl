import torch
import torch.nn as nn
import torch.optim as optim
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

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

# Hyperparameter configuration
config = {
    'input_size': 5,
    'num_layers': 2,
    'hidden_size': 128,
    'output_size': 1,
    'learning_rate': 0.001,
    'batch_size': 128,
    'num_epochs': 200
}

# Load training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert pandas DataFrames to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# Instantiate the model
model = NeuralNetwork(**config)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = model.configure_optimizers()

# Train the model
train_model(model, train_loader, criterion, optimizer, config['num_epochs'])

# Save the trained model
save_model(model, 'trained_model.pth')
