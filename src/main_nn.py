import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Define a more general neural network
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

def predict(model, input_data):
    with torch.no_grad():
        predictions = model(input_data)
    return predictions.numpy()

def main():
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

    # Instantiate the model
    model = NeuralNetwork(**config)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = model.configure_optimizers()

    for season in range(2016, 2022):  # Update the range to cover the desired seasons
        for gameweek in range(1, 39):  # There are 38 gameweeks in a season
            # Load training data
            train_data = load_data(season, gameweek)

            # Select features and target variable
            features = ['assists', 'minutes', 'goals_scored', 'ict_index', 'threat']
            target = 'total_points'

            X = train_data[features]
            y = train_data[target]

            # Convert pandas DataFrames to PyTorch tensors
            X_train_tensor = torch.tensor(X.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y.values, dtype=torch.float32)

            # Create DataLoader for training
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

            # Train the model
            train_model(model, train_loader, criterion, optimizer, config['num_epochs'])

    # Now, you can predict player total point scores for the 2022/23 season
    # Load testing data for the 2022/23 season
    test_data = load_data(2022, 1)  # You can adjust gameweek as needed

    # Select features for testing
    X_test = test_data[features]

    # Convert testing data to PyTorch tensor
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    # Make predictions on the test set
    predictions = predict(model, X_test_tensor)

    # Convert predictions back to a numpy array
    print(predictions)

if __name__ == "__main__":
    main()
