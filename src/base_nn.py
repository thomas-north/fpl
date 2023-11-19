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

# Define an improved neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Instantiate the improved model
input_size = X_train.shape[1]
model = NeuralNetwork(input_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the improved neural network
num_epochs = 100  # Increase the number of epochs
for epoch in range(num_epochs):
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

# Evaluate the improved neural network
mse_nn = mean_squared_error(y_test, y_pred_nn)
print(f'Improved Neural Network Mean Squared Error: {mse_nn}')
