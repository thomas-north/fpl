import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), '../model/trained_model.pth')

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
        'num_epochs': 100
    }

    # Instantiate the model
    model = NeuralNetwork(**config)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = model.configure_optimizers()

    for season in tqdm(range(2021, 2022), desc='Seasons'):
        for gameweek in tqdm(range(1, 39), desc=f'{season}-{season+1} Gameweeks', leave=False):
            # Load training data
            szn = '{}'.format(str(season)+'-'+str(season+1)[2:])
            train_data = load_data(szn, gameweek)

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

    # Test the model against all gameweeks of the 2022/23 season
    test_season = f'2022-23'
    test_data = pd.concat([load_data(test_season, gameweek) for gameweek in range(1, 39)])

    # Select features for testing
    features = ['assists', 'minutes', 'goals_scored', 'ict_index', 'threat']
    target = 'total_points'

    X_test = test_data[features]
    y_test = test_data[target]

    # Handle missing values
    X_test['assists'].fillna(0, inplace=True)
    X_test['minutes'].fillna(0, inplace=True)
    X_test['goals_scored'].fillna(0, inplace=True)

    # Convert to numeric
    X_test['assists'] = pd.to_numeric(X_test['assists'], errors='coerce')
    X_test['minutes'] = pd.to_numeric(X_test['minutes'], errors='coerce')
    X_test['goals_scored'] = pd.to_numeric(X_test['goals_scored'], errors='coerce')

    # Verify data types
    print(X_test.dtypes)

    # Convert testing data to PyTorch tensor
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    #y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Make predictions on the test set
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

    # Convert predictions back to a numpy array
    y_pred_nn = y_pred_tensor.numpy()

    # Evaluate the model on the test set
    mse_test = mean_squared_error(y_test, y_pred_nn)
    print(f'Mean Squared Error on Test Set: {mse_test}')

if __name__ == "__main__":
    main()
