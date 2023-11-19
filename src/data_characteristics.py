import pandas as pd
import matplotlib.pyplot as plt

dataset = '../data/2016-17/gws/gw1.csv'

data = pd.read_csv(dataset) 

# Display basic information about the dataset
print(data.info())

# Display the first few rows of the dataset
print(data.head())

# Summary statistics for numerical columns
print(data.describe())

# Unique values in categorical columns
print(data['position'].unique())  # Adjust column names as needed

# Distribution of total points
plt.hist(data['total_points'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Points')
plt.xlabel('Total Points')
plt.ylabel('Frequency')
plt.show()
