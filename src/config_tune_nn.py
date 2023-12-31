# A neural network focusing on one player
# In this case: Mohamed Salah

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import fuzzywuzzy
import itertools
import numpy as np

from fuzzywuzzy import process
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

SHOW_PLOTS = True

# Define a neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size, learning_rate, batch_size, num_epochs):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(input_size if not self.layers else hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

def load_data(season, player_name):
    # Specify the path to the players' data
    players_data_path = f'../data/{season}/players/'

    # Get a list of all player names in the specified season directory
    available_players = [player.replace('_', ' ') for player in os.listdir(players_data_path) if os.path.isdir(os.path.join(players_data_path, player))]

    # Use fuzzy matching to find the closest match for the given player_name
    closest_match, score = process.extractOne(player_name, available_players)

    # Construct the path to the player's data based on the closest match
    player_path = os.path.join(players_data_path, closest_match.replace(' ', '_'))

    # Load the player's data
    dataset = os.path.join(player_path, 'gw.csv')
    data = pd.read_csv(dataset, encoding="ISO-8859-1")

    # Feature engineering and scaling can be done here
    return data

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            if inputs.size(0) < 8:
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

def grid_search(player_name, param_grid, features, target):
    best_params = None
    best_mse = float('inf')  # Set to positive infinity initially

    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*param_grid.values()))

    for params in param_combinations:
        # Convert params to a dictionary
        param_dict = dict(zip(param_grid.keys(), params))

        # Add missing input_size and output_size to param_dict
        param_dict['input_size'] = len(features)
        param_dict['output_size'] = 1

        # Instantiate the model
        model = NeuralNetwork(**param_dict)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = model.configure_optimizers()

        # Load training data
        train_data = pd.concat([load_data('{}'.format(str(season)+'-'+str(season+1)[2:]), player_name) for season in range(2017, 2022)])

        X = train_data[features]
        y = train_data[[target]]

        # Handle missing or non-numeric values in X
        X = X.apply(pd.to_numeric, errors='coerce')

        # Drop rows with missing values
        X = X.dropna()

        X.reset_index(drop=True, inplace=True)
        y = y.reset_index(drop=True)

        # Convert pandas DataFrames to PyTorch tensors
        X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y.values, dtype=torch.float32)

        # Create DataLoader for training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True)

        try:
            # Train the model
            train_model(model, train_loader, criterion, optimizer, param_dict['num_epochs'])

            # Evaluate the model on the test set
            test_data = load_data('2022-23', player_name)
            X_test = test_data[features]
            y_test = test_data[target]

            # Handle missing values
            X_test = X_test.apply(pd.to_numeric, errors='coerce')
            X_test = X_test.dropna()

            # Convert testing data to PyTorch tensor
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

            # Make predictions on the test set
            with torch.no_grad():
                y_pred_tensor = model(X_test_tensor)

            # Convert predictions back to a numpy array
            y_pred_nn = y_pred_tensor.numpy()

            # Evaluate the model
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
            mse_test = mean_squared_error(y_test_tensor, y_pred_nn)

            # Update best parameters if the current combination is better
            if mse_test < best_mse:
                best_mse = mse_test
                best_params = param_dict

        except Exception as e:
            print(f"Skipping iteration due to an error: {e}")

    return best_params, best_mse


def main():
    # Tune model to a player
    player_name = 'Mohamed_Salah'

    # Select features and target variable
    features = ['assists', 'minutes', 'goals_scored', 'ict_index', 'threat']
    target = 'total_points'

    # Hyperparameter grid for grid search
    config = {
        'input_size': [len(features)],
        'num_layers': [4, 5, 6],
        'hidden_size': [64, 128, 256],
        'output_size': [1],
        'learning_rate': [0.001, 0.01, 0.05],
        'batch_size': [64, 128, 256],
        'num_epochs': [100, 200, 300],
    }

    # Use grid search to find the best combination of hyperparameters
    best_params, best_mse = grid_search(player_name, config, features, target)
    print("\n", best_params, best_mse)

    # Update the config dictionary with the best parameters
    config.update(best_params)

    # Train the model with the best parameters
    model = NeuralNetwork(**config)
    criterion = nn.MSELoss()
    optimizer = model.configure_optimizers()

    # Load training data for Mo Salah
    train_data = pd.concat([load_data('{}'.format(str(season)+'-'+str(season+1)[2:]), player_name) for season in range(2017, 2022)])

    X = train_data[features]
    y = train_data[[target]] 

    # Handle missing or non-numeric values in X
    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    X = X.dropna()

    X.reset_index(drop=True, inplace=True)
    y = y.reset_index(drop=True)

    # Convert pandas DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, config['num_epochs'])

    # Generate predictions for the 2022-23 season
    test_season = '2022-23'
    test_data = load_data(test_season, player_name)

    # Select features for testing
    X_test = test_data[features]
    y_test = test_data[target]

    # Handle missing values
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.dropna()

    # Convert testing data to PyTorch tensor
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    # Make predictions on the test set
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

    # Convert predictions back to a numpy array
    y_pred_nn = y_pred_tensor.numpy()

    # Save the trained model
    torch.save(model.state_dict(), '../model/trained_tuned_model.pth')

    # Evaluate the model on the test set
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Check for NaN values in predictions and true labels
    if torch.isnan(y_pred_tensor).any().item() or torch.isnan(y_test_tensor).any().item():
        print("NaN values found in predictions or true labels. Please handle missing values.")

    else:
        # Calculate mean squared error if no NaN values are present
        mse_test = mean_squared_error(y_test_tensor, y_pred_nn)
        total_points_actual = y_test.sum()
        total_points_predicted = y_pred_tensor.sum()
        print(f'Mean Squared Error on Test Set: {mse_test}')
        print(f'Total points actual: {total_points_actual:.0f}')
        print(f'Total points predicted: {total_points_predicted:.0f}')

        if SHOW_PLOTS == True:
            # Plot predictions vs actual outcomes (you can customize this plot based on your needs)
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(y_test_tensor, label='Actual Total Points')
            plt.plot(y_pred_nn, label='Predicted Total Points')
            plt.title(f'Actual vs Predicted Total Points\nTotal Points (Actual): {total_points_actual:.0f}, Total Points (Predicted): {total_points_predicted:.0f}')
            plt.xlim(0,37)
            plt.xlabel('Gameweek')
            plt.ylabel('Total Points')
            plt.legend()
            plt.show()

if __name__ == "__main__":
    main()
