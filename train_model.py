import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r"C:\Users\sweth\OneDrive\Desktop\final project\madurai_housing_dataset_with_price.csv")  # Load the housing dataset

# Select features (independent variables) and target (dependent variable)
X = df[['Square Feet', 'No of Bedrooms', 'Bathrooms', 'Age of House', 'Base Price per Sqft']]  # Features used for prediction
y = df['Estimated Price']  # Target variable (House Price)

# Perform One-Hot Encoding for categorical variables like 'Area' if present in the dataset
X = pd.get_dummies(X)  # Converts categorical columns into multiple binary columns

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Create a directory named 'models' if it does not already exist
os.makedirs("models", exist_ok=True)

# Save the trained model using Pickle for later use
pickle.dump(model, open("models/house_price_model.pkl", "wb"))

# Save the column names of the dataset to ensure consistency in future predictions
pickle.dump(X.columns.tolist(), open("models/columns.pkl", "wb"))

print("Model and column names saved successfully!")  # Confirmation message
