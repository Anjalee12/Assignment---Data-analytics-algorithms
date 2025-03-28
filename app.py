from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load("optimized_kmeans_iris.joblib")

# Initialize Flask app
app = Flask(__name__)

# Standard scaler (important to scale input like in training)
scaler = StandardScaler()

# Root route
@app.route('/')
def home():
    return "Welcome to the K-Means API! Use the /predict endpoint for single inputs or /predict_csv for CSV uploads."

# Define API endpoint for single predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Standardize input (same as model training)
        features_scaled = scaler.fit_transform(features)

        # Predict cluster
        cluster = model.predict(features_scaled)[0]
        return jsonify({"cluster": int(cluster)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Define API endpoint for CSV file predictions
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # Get the uploaded file
        file = request.files['file']
        df = pd.read_csv(file)

        # Drop non-numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        # Standardize input (same as model training)
        features_scaled = scaler.fit_transform(df_numeric)

        # Predict clusters
        clusters = model.predict(features_scaled)
        df['cluster'] = clusters

        # Convert to JSON and return
        return df.to_json(orient='records')
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
#if __name__ == '__main__':
    #app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000
    app.run(host="0.0.0.0", port=port)
