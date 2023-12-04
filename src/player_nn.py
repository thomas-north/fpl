# A neural network focusing on one player
# In this case: Mohamed Salah

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from fuzzywuzzy import process  # Add this import statement
from sklearn.preprocessing import StandardScaler

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

# Function to calculate fixture difficulty
def calculate_fixture_difficulty(row, fixtures_data):
    fixture = fixtures_data[fixtures_data['id'] == row['fixture']]
    if not fixture.empty:
        if row['was_home']:
            return fixture['team_h_difficulty'].values[0]
        else:
            return fixture['team_a_difficulty'].values[0]
    else:
        return None

# Modify the load_data function
def load_data(season, player_name, fixtures_data):
    # Specify the path to the players' data
    players_data_path = f'../data/{season}/players/'

    # Get a list of all player names in the specified season directory
    available_players = [player.replace('_', ' ') for player in os.listdir(players_data_path) if os.path.isdir(os.path.join(players_data_path, player))]

    # Use fuzzy matching to find the closest match for the given player_name
    closest_match, _ = process.extractOne(player_name, available_players)

    # Construct the path to the player's data based on the closest match
    player_path = os.path.join(players_data_path, closest_match.replace(' ', '_'))

    # Load the player's data
    dataset = os.path.join(player_path, 'gw.csv')
    data = pd.read_csv(dataset, encoding="ISO-8859-1")

    # Calculate fixture difficulty
    data['fixture_difficulty'] = data.apply(lambda row: calculate_fixture_difficulty(row, fixtures_data), axis=1)

    # Feature engineering and scaling can be done here

    # Extract season from kickoff_time and create a new 'season' column
    data['season'] = pd.to_datetime(data['kickoff_time']).dt.year

    data['week_of_season'] = data.groupby('season').cumcount() + 1

    return data



# Function to train the model
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

    # Save the trained model
    torch.save(model.state_dict(), '../model/trained_player_model.pth')

# Modify the main function
def main():

    # Select features and target variable
    player_name = 'Mohamed_Salah'
    features = ['ict_index', 'threat', 'was_home']
    target = 'total_points'

    # Hyperparameter configuration
    config = {
        'input_size': len(features) + 2,  # Add 2 for contextual features
        'num_layers': 6,
        'hidden_size': 128,
        'output_size': 1,
        'learning_rate': 0.01,
        'batch_size': 32,
        'num_epochs': 200
    }

    # Instantiate the model
    model = NeuralNetwork(**config)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = model.configure_optimizers()

    # Load fixtures data for the specified seasons
    fixtures_data = pd.concat([pd.read_csv('../data/{}/fixtures.csv'.format(str(season)+'-'+str(season+1)[2:])) for season in range(2018, 2022)])

    # Load training data for Mo Salah
    train_data = pd.concat([load_data('{}'.format(str(season)+'-'+str(season+1)[2:]), player_name, fixtures_data) for season in range(2018, 2022)])

    # Introduce new contextual features
    train_data['week_of_season'] = train_data.index + 1

    # Calculate fixture difficulty for the training data
    train_data['fixture_difficulty'] = train_data.apply(lambda row: calculate_fixture_difficulty(row, fixtures_data), axis=1)

    # Remove future-dependent features
    future_features = ['assists', 'goals_scored', 'minutes']
    train_data.drop(columns=future_features, inplace=True)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

    X_train = train_data[features + ['fixture_difficulty', 'week_of_season']]
    y_train = train_data[[target]]

    X_test = test_data[features + ['fixture_difficulty', 'week_of_season']]
    y_test = test_data[[target]]

    # Handle missing or non-numeric values in X
    X_train['was_home'] = X_train['was_home'].astype(int)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_train = X_train.dropna()

    X_test['was_home'] = X_test['was_home'].astype(int)
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.dropna()

    # Convert pandas DataFrames to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, config['num_epochs'])

    # Generate predictions for the upcoming season
    # Use a similar approach as for training data to introduce contextual features
    test_season = '2022-23'
    test_data = load_data(test_season, player_name, fixtures_data)

    test_data['fixture_difficulty'] = test_data.apply(lambda row: calculate_fixture_difficulty(row, fixtures_data), axis=1)

    # Calculate 'week_of_season' based on the index of rows within each season
    test_data['week_of_season'] = test_data.groupby('season').cumcount() + 1

    # Select features for testing
    selected_features = ['ict_index', 'threat', 'was_home', 'fixture_difficulty', 'week_of_season']
    X_test_upcoming = test_data[selected_features]
    y_test_upcoming = test_data[[target]]

    # Handle missing values
    X_test_upcoming['was_home'] = X_test_upcoming['was_home'].astype(int)
    X_test_upcoming = X_test_upcoming.apply(pd.to_numeric, errors='coerce')
    X_test_upcoming = X_test_upcoming.dropna()

    # Convert testing data to PyTorch tensor
    X_test_upcoming_tensor = torch.tensor(X_test_upcoming.values, dtype=torch.float32)

    # Make predictions on the upcoming season
    with torch.no_grad():
        y_pred_upcoming_tensor = model(X_test_upcoming_tensor)

    # Convert predictions back to a numpy array
    y_pred_upcoming_nn = y_pred_upcoming_tensor.numpy()


    # Evaluate the model on the test set
    y_test_upcoming_tensor = torch.tensor(y_test_upcoming.values, dtype=torch.float32)

    mse_test_upcoming = mean_squared_error(y_test_upcoming_tensor, y_pred_upcoming_nn)
    total_points_actual_upcoming = y_test_upcoming.sum()
    total_points_predicted_upcoming = y_pred_upcoming_tensor.sum()
    print(f'\nMean Squared Error on Test Set for Upcoming Season: {mse_test_upcoming}')
    print(f'Total points actual (Upcoming Season): {total_points_actual_upcoming}')
    print(f'Total points predicted (Upcoming Season): {total_points_predicted_upcoming}')

    # Create DataLoader for testing
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Evaluate the model on the test set
    def evaluate_model(model, test_loader, criterion):
        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                test_loss += loss.item()

        average_test_loss = test_loss / len(test_loader)
        return average_test_loss

    # Calculate and print the average test loss
    avg_test_loss = evaluate_model(model, test_loader, criterion)
    print(f'Average Test Loss: {avg_test_loss}\n')

    # Make predictions on the test set
    with torch.no_grad():
        y_pred_test_tensor = model(X_test_tensor)

    # Convert predictions back to a numpy array
    y_pred_test_nn = y_pred_test_tensor.numpy()

    if SHOW_PLOTS:
        # Plot predictions vs actual outcomes (you can customize this plot based on your needs)
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_upcoming_tensor, label='Actual Total Points')
        plt.plot(y_pred_upcoming_nn, label='Predicted Total Points')
        plt.title(f'Actual vs Predicted Total Points for Upcoming Season\nTotal Points (Actual): {total_points_actual_upcoming}, Total Points (Predicted): {total_points_predicted_upcoming}')
        plt.xlabel('Gameweek')
        plt.ylabel('Total Points')
        plt.legend()
        plt.show()

        # Visualize predictions vs actual outcomes on the test set
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_tensor, label='Actual Total Points')
        plt.plot(y_pred_test_nn, label='Predicted Total Points')
        plt.title('Actual vs Predicted Total Points on Test Set')
        plt.xlabel('Data Point')
        plt.ylabel('Total Points')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
