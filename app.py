from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os  # Import os for port setting
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load("optimized_kmeans_iris.joblib")

# Initialize Flask app
app = Flask(__name__)

# Initialize scaler with training statistics (replace with actual values from your dataset)
scaler = StandardScaler()
scaler.mean_ = np.array([5.8, 3.0, 3.7, 1.2])  # Replace with actual training mean
scaler.scale_ = np.array([0.8, 0.4, 1.7, 0.7])  # Replace with actual training std

# Root route (GET allowed)
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the K-Means API! Use the /predict endpoint for single inputs or /predict_csv for CSV uploads."

# Define API endpoint for single predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Standardize input (use transform, not fit_transform)
        features_scaled = scaler.transform(features)

        # Predict cluster
        cluster = model.predict(features_scaled)[0]
        return jsonify({"cluster": int(cluster)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Define API endpoint for CSV file predictions
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # Get the uploaded file
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file)

        # Drop non-numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        # Check for missing values
        if df_numeric.isnull().values.any():
            return jsonify({"error": "CSV contains missing values"}), 400

        # Standardize input (use transform, not fit_transform)
        features_scaled = scaler.transform(df_numeric)

        # Predict clusters
        clusters = model.predict(features_scaled)
        df['cluster'] = clusters

        # Convert to JSON and return
        return df.to_json(orient='records')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000
    app.run(host="0.0.0.0", port=port)
