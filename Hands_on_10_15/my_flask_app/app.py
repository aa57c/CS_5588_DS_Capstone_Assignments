from flask import Flask, request, jsonify
import networkx as nx
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import requests
import boto3
import io
from keras.layers import TFSMLayer


# Initialize a session using your AWS credentials
s3 = boto3.client('s3')

# Specify your bucket name and file names
bucket_name = 'ds-capstone-hands-on-10-15'
csv_file_key = 'merged_CGM_clinical_data.csv'
keras_file_key = 'best_dl_model_female.keras'

# Load the CSV file
csv_obj = s3.get_object(Bucket=bucket_name, Key=csv_file_key)
csv_data = csv_obj['Body'].read().decode('utf-8')
CGM_data = pd.read_csv(io.StringIO(csv_data))

# Load the Keras model from S3
keras_obj = s3.get_object(Bucket=bucket_name, Key=keras_file_key)
keras_data = keras_obj['Body'].read()

# Save the model temporarily to load it using Keras
with open('temp_model.keras', 'wb') as f:
    f.write(keras_data)

# Now load the model using Keras
model = load_model('temp_model.keras')

# Remove every column besides "Hora", "Glucemia", and "T2DM". Make sure to only include the female samples (gender = 0)
CGM_data = CGM_data[CGM_data["gender"] == 0]
CGM_data = CGM_data[["Hora", "Glucemia", "T2DM"]]

# Label encode T2DM: 0 - False, 2 - True
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
CGM_data["T2DM"] = encoder.fit_transform(CGM_data["T2DM"])

CGM_data = CGM_data.rename(columns={"T2DM": "Diabetes_Status"})

# Preprocess data
CGM_data['Hora'] = pd.to_timedelta(CGM_data['Hora']).dt.total_seconds()
CGM_data = CGM_data.dropna()

# Define numerical columns
numerical_columns = ['Hora', 'Glucemia']

# Normalize numerical columns
scaler = MinMaxScaler()
CGM_data[numerical_columns] = scaler.fit_transform(CGM_data[numerical_columns])

# Handle class imbalance by oversampling the minority class
minority_class = CGM_data[CGM_data['Diabetes_Status'] == 1]
majority_class = CGM_data[CGM_data['Diabetes_Status'] == 0]

minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
balanced_data = pd.concat([majority_class, minority_upsampled])

# Prepare the features and target
X_data = balanced_data[numerical_columns].values
y_data = balanced_data['Diabetes_Status'].values

# Set the number of timesteps (sliding window)
window_size = 10

# Define a function to create sequences of data (sliding window approach)
def create_sequences(data, target, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(target[i+window_size])  # Predict the value after the window
    return np.array(X), np.array(y)

# Create sequences using reduced features
X, y = create_sequences(X_data, y_data, window_size)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to predict diabetes status using the model
def predict_diabetes(cgm_sequence):
    """
    Predict diabetes status given a sequence of glucose data (CGM data).

    Parameters:
    - cgm_sequence: A numpy array of shape (timesteps, 2) where 2 corresponds to 'Hora' and 'Glucemia'.
    
    Returns:
    - A string indicating whether diabetes is detected or not.
    """
    # Preprocess the input
    cgm_sequence = scaler.transform(cgm_sequence)  # Normalize the input sequence
    
    # Reshape to match the model's expected input
    cgm_sequence = np.expand_dims(cgm_sequence, axis=0)  # Add batch dimension
    
    # Get the model's prediction
    prediction = (model.predict(cgm_sequence) > 0.5).astype(int)
    
    # Return the result based on the prediction
    return "Diabetes detected" if prediction[0][0] == 1 else "No Diabetes detected"

# Example usage with a test sequence from X_test
example_sequence = X_test[0]  # A sequence from your test set
result = predict_diabetes(example_sequence)
print("Prediction result:", result)

# Flask App setup
app = Flask(__name__)

# Create a new graph
G = nx.Graph()

# Add nodes for diabetes classes
G.add_node("Prediabetes", type="condition")
G.add_node("Type 2 Diabetes", type="condition")
G.add_node("Gestational Diabetes", type="condition")
G.add_node("No Diabetes", type="condition")

# Add nodes for management techniques
G.add_node("Healthy Diet", type="treatment")
G.add_node("Exercise", type="treatment")
G.add_node("Blood Glucose Monitoring", type="treatment")
G.add_node("Oral Medication", type="treatment")
G.add_node("Insulin Therapy", type="treatment")
G.add_node("Prenatal Care", type="treatment")
G.add_node("Lifestyle Changes", type="treatment")

# Add edges (relationships) for each diabetes class to its management techniques
# Prediabetes
G.add_edge("Prediabetes", "Healthy Diet")
G.add_edge("Prediabetes", "Exercise")
G.add_edge("Prediabetes", "Lifestyle Changes")
G.add_edge("Prediabetes", "Blood Glucose Monitoring")

# Type 2 Diabetes
G.add_edge("Type 2 Diabetes", "Oral Medication")
G.add_edge("Type 2 Diabetes", "Insulin Therapy")
G.add_edge("Type 2 Diabetes", "Healthy Diet")
G.add_edge("Type 2 Diabetes", "Exercise")
G.add_edge("Type 2 Diabetes", "Blood Glucose Monitoring")

# Gestational Diabetes
G.add_edge("Gestational Diabetes", "Prenatal Care")
G.add_edge("Gestational Diabetes", "Blood Glucose Monitoring")
G.add_edge("Gestational Diabetes", "Healthy Diet")
G.add_edge("Gestational Diabetes", "Exercise")

# No Diabetes (for prevention)
G.add_edge("No Diabetes", "Healthy Diet")
G.add_edge("No Diabetes", "Exercise")

# Function to retrieve treatments from the knowledge graph based on disease prediction
def get_treatments(disease):
    treatments = [n for n in G.neighbors(disease) if G.nodes[n]['type'] == 'treatment']
    return treatments

import matplotlib.pyplot as plt
def visualize_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10)
    plt.show()

# Visualize the knowledge graph
visualize_graph(G)

# Flask route to handle prediction and return recommendations
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cgm_sequence = np.array(data["cgm_sequence"])

    # Reshape the input if needed (assumes a 2D array with 'Hora' and 'Glucemia')
    if cgm_sequence.shape[1] != 2:
        return jsonify({"error": "Invalid input shape, expected 2 columns: 'Hora' and 'Glucemia'."})

    # Predict diabetes status
    prediction = predict_diabetes(cgm_sequence)

    # If diabetes is detected, query the knowledge graph for treatments
    if prediction == "Diabetes detected":
        treatments = get_treatments("Type 2 Diabetes")
    else:
        treatments = []

    # Return prediction and recommendations as JSON
    return jsonify({
        "prediction": prediction,
        "recommendations": treatments
    })

# Run Flask server
if __name__ == "__main__":
    app.run(port=5000)
