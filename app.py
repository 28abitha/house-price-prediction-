from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained machine learning model from the saved file
model = pickle.load(open("models/house_price_model.pkl", "rb"))

# Load the column names to maintain consistency in feature encoding
columns = pickle.load(open("models/columns.pkl", "rb"))

# Define route for the homepage
@app.route("/")
def home():
    return render_template("homepage.html")  # Renders the homepage HTML file

# Define route for the login page
@app.route("/login")
def login():
    return render_template("login.html")  # Renders the login page HTML file

# Define route for the register page
@app.route("/register")
def register():
    return render_template("register.html")  # Renders the registration page HTML file

# Define route for house price prediction (POST method)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON data from the client-side request
        data = request.get_json()
        
        # Extract input features from the request
        square_feet = float(data["square_feet"])
        bedrooms = int(data["bedrooms"])
        bathrooms = int(data["bathrooms"])
        age = int(data["age"])
        base_price_sqft = float(data["base_price_sqft"])
        area = data["area"]

        # Create a DataFrame for user input with the required column structure
        user_data = pd.DataFrame([[square_feet, bedrooms, bathrooms, age, base_price_sqft]], 
                                 columns=['Square Feet', 'No of Bedrooms', 'Bathrooms', 'Age of House', 'Base Price per Sqft'])

        # Perform One-Hot Encoding for 'Area' column to match the model's input features
        for col in columns:
            if col.startswith("Area_"):
                user_data[col] = 1 if f"Area_{area}" == col else 0

        # Reorder columns to match the trained model's structure, filling missing values with 0
        user_data = user_data.reindex(columns=columns, fill_value=0)

        # Predict house price using the trained model
        predicted_price = model.predict(user_data)[0]

        # Return the predicted price as JSON response
        return jsonify({"predicted_price": round(predicted_price, 2)})

    except Exception as e:
        # Handle errors and return error message as JSON response
        return jsonify({"error": str(e)})

# Run the Flask app in debug mode
if __name__ == "__main__":
    app.run(debug=True)
    