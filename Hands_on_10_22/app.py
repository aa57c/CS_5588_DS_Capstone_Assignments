import tensorflow as tf
import numpy as np
import streamlit as st
from transformers import pipeline
import pickle
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

scaler = MinMaxScaler()

model = tf.keras.models.load_model('my_model.h5')
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

svd = joblib.load('svd_model.joblib')


# Load a text generation model for recommendations
recommendation_generator = pipeline("text-generation", model="openai-community/gpt2")

# Function to preprocess data and create sequences
def create_sequences_for_patient(df, window_size, numerical_cols):
    sequences = []
    
    grouped_df = df.groupby('Patient_ID')
    for _, group in grouped_df:
        group = group.sort_values('Hora')
        X_data = group[numerical_cols].values
        
        for i in range(len(group) - window_size):
            sequences.append(X_data[i:i + window_size])
            
    return np.array(sequences)

def preprocess_data(df):
  # drop target column
  df = df.drop(columns=['T2DM'])
  df['Hora'] = pd.to_timedelta(df['Hora']).dt.total_seconds()
  
  # Scale the numerical columns
  df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
  return df

def reduce_dimensions(sequences):
  n_components = 5  # Reduce features to 5 dimensions
  sequences_reduced = svd.transform(sequences.reshape(-1, sequences.shape[2]))
  sequences_reduced = sequences_reduced.reshape(sequences.shape[0], sequences.shape[1], n_components)
  return sequences_reduced

def generate_recommendation(diagnosis):
  # Generate a recommendation based on the diagnosis
  prompt = ""
  if diagnosis == 0:
    prompt = "Provide health recommendations for a person diagnosed with no diabetes."
  elif diagnosis == 1:
    prompt = "Provide health recommendations for a person diagnosed with type 2 diabetes."
  recommendation = recommendation_generator(prompt, max_length=200, num_return_sequences=1)
  return recommendation[0]['generated_text']

def predict_diabetes_and_recommend(reduced_sequences):
  predictions = model.predict(reduced_sequences)
  predicted_class = np.argmax(predictions, axis=1)[0]
  # Generate a recommendation based on the diagnosis
  recommendation = generate_recommendation(predicted_class)
    
  return predicted_class, recommendation



# Streamlit app interface
st.title("CGM Data Diabetes Prediction")

# Upload CGM data
uploaded_file = st.file_uploader("Choose a CSV file with CGM data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CGM Data:")
    st.write(df.head())

    # Preprocess and create sequences
    window_size = 10
    # Define numerical columns
    numerical_cols = ['Hora', 'Glucemia', 'BMI', 'age', 'HbA1c', 'follow.up']

    preprocesed_df = preprocess_data(df)
    sequences = create_sequences_for_patient(preprocesed_df, window_size, numerical_cols)
    reduced_sequences = reduce_dimensions(sequences)

    prediction, recommendation = predict_diabetes_and_recommend(reduced_sequences)

    # Display the diagnosis
    if prediction == 0:
        st.write("Diagnosis: No Diabetes")
    elif prediction == 1:
        st.write("Diagnosis: Type 2 Diabetes")
    
    # Display the recommendation
    st.write("Health Recommendation:")
    st.write(recommendation)