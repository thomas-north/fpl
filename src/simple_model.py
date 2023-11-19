import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = '../data/2016-17/gws/gw1.csv'

data = pd.read_csv(dataset, encoding = "ISO-8859-1") 

# Example: Create a new feature 'goals_per_minute'
data['goals_per_minute'] = data['goals_scored'] / data['minutes']

# Example: Scale numerical features to ensure they have a similar range. This is important for neural networks.
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

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize predictions vs. actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Total Points')
plt.ylabel('Predicted Total Points')
plt.title('Predictions vs. Actual Values')
plt.show()
