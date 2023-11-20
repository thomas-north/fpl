import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Get data
dataset = '../data/2016-17/gws/gw1.csv'
data = pd.read_csv(dataset, encoding="ISO-8859-1")

# Feature Engineering
data['goals_per_minute'] = data['goals_scored'] / data['minutes']

# Feature Scaling
scaler = StandardScaler()
numerical_cols = ['assists', 'minutes', 'goals_scored', 'ict_index', 'threat', 'total_points']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Select features and target variable
features = ['assists', 'minutes', 'goals_scored', 'ict_index', 'threat']
target = 'total_points'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert pandas DataFrames to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

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

# Hyperparameter configuration
config = {
    'input_size': X_train.shape[1],
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

# Train the neural network
for epoch in range(config['num_epochs']):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

# Make predictions on the test set
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

# Convert predictions back to a numpy array
y_pred_nn = y_pred_tensor.numpy()

# Evaluate the neural network
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Neural Network Mean Squared Error: {mse_nn}')
