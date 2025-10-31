# CS50_Crypto_Predictor.py
# Author: Muhammad Faiq Hayat
# Description:
# A mini data science project analyzing cryptocurrency market trends
# and predicting Bitcoin closing prices using a linear regression model.
# Created as part of an independent data analytics exploration.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load and inspect the dataset
df = pd.read_csv('crypto_data_updated_13_november.csv')
print("Dataset shape:", df.shape)
print(df.head())

# Convert Date column and extract temporal features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Drop original Date column for modeling
df.drop('Date', axis=1, inplace=True)

# Define input (X) and target (y)
X = df.drop('Close (BTC)', axis=1)
y = df['Close (BTC)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and linear regression
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

# Predict and evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Training R² Score: {r2_train:.4f}")
print(f"Testing R² Score: {r2_test:.4f}")

# Visualization of Actual vs Predicted BTC Prices
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.title("Actual vs Predicted Bitcoin Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True)
plt.show()

print("\nProject successfully completed — model trained and evaluated.")
