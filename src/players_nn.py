import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from fuzzywuzzy import process
import logging

SHOW_PLOTS = True
LOG_FILE = '../logs/training.log'

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

    # Load the player's data if available, otherwise return None
    dataset = os.path.join(player_path, 'gw.csv')
    if os.path.exists(dataset):
        data = pd.read_csv(dataset, encoding="ISO-8859-1")
        return data
    else:
        return None

def train_model(model, train_loader, criterion, optimizer, num_epochs, player_name):
    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - {player_name}', leave=False):
            if inputs.size(0) < 8:
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

    # Save the trained model
    model_save_path = f'../model/{player_name}_trained_model.pth'
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model for {player_name} saved at {model_save_path}")

def main():
    # Select features and target variable
    features = ['assists', 'minutes', 'goals_scored', 'ict_index', 'threat']
    target = 'total_points'

    # Hyperparameter configuration
    config = {
        'input_size': len(features),
        'num_layers': 6,
        'hidden_size': 128,
        'output_size': 1,
        'learning_rate': 0.01,
        'batch_size': 32,
        'num_epochs': 200
    }

    # List of players for training
    players_for_training = ['Mohamed_Salah', 'Harry_Kane']

    # List of players for testing
    players_for_testing = ['Mohamed_Salah', 'Harry_Kane']

    # Training phase
    for player_name in players_for_training:
        # Instantiate the model
        model = NeuralNetwork(**config)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = model.configure_optimizers()

        # Load training data for the current player
        train_data = pd.concat([load_data('{}'.format(str(season)+'-'+str(season+1)[2:]), player_name) for season in range(2017, 2022) if load_data('{}'.format(str(season)+'-'+str(season+1)[2:]), player_name) is not None])

        if train_data is None:
            logging.warning(f"Skipping training for {player_name}. Data not available for all seasons.")
            continue

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
        train_model(model, train_loader, criterion, optimizer, config['num_epochs'], player_name)

    # Testing phase
    for player_name in players_for_testing:
        # Instantiate the model
        model = NeuralNetwork(**config)

        # Load the trained model for the current player if available
        model_path = f'../model/{player_name}_trained_model.pth'

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

            # Load test data for the current player
            test_data = load_data('2022-23', player_name)

            if test_data is not None:
                # Case (1): We have historic data for the player
                X_test = test_data[features]
                y_test = test_data[target]
                
            else:
                # Case (2): New player, generate synthetic input features
                X_test = pd.DataFrame({'assists': [0], 'minutes': [90], 'goals_scored': [0], 'ict_index': [0], 'threat': [0]})
                y_test = pd.DataFrame({'total_points': [0]})

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

            # Evaluate the model on the test set
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

            mse_test = mean_squared_error(y_test_tensor, y_pred_nn)
            total_points_actual = y_test.sum()
            total_points_predicted = y_pred_tensor.sum()
            logging.info(f'Mean Squared Error for {player_name}: {mse_test}')
            logging.info(f'Total points actual: {total_points_actual:.0f}')
            logging.info(f'Total points predicted: {total_points_predicted:.0f}')

            if SHOW_PLOTS:
                logging.debug("Reached the plotting section")
                # Plot predictions vs actual outcomes
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 6))
                plt.plot(y_test_tensor, label='Actual Total Points')
                plt.plot(y_pred_nn, label='Predicted Total Points')
                plt.title(f'Actual vs Predicted Total Points for {player_name}\nTotal Points (Actual): {total_points_actual:.0f}, Total Points (Predicted): {total_points_predicted:.0f}')
                plt.xlabel('Gameweek')
                plt.ylabel('Total Points')
                plt.legend()
                plt.show()

if __name__ == "__main__":
    main()
