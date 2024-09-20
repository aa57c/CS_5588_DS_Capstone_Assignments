import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from flask import Flask, request, jsonify
import streamlit as st

# Step 1: Initialize Hugging Face Motion Detection Model
# Use a valid Hugging Face model for image classification (e.g., 'google/vit-base-patch16-224')
motion_model = pipeline('image-classification', model='google/vit-base-patch16-224')

# Step 2: Convert OpenCV frame to PIL image for Hugging Face model
def detect_motion(frame):
    # Convert OpenCV BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL image
    pil_image = Image.fromarray(rgb_frame)
    
    # Use Hugging Face's model for image classification
    return motion_model(pil_image)

# Step 3: Build a Custom Internal Model for Event Prediction
def build_event_prediction_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Assuming some preprocessed data for training
X_train = np.random.rand(100, 10)  # Dummy data
y_train = np.random.randint(2, size=100)  # Dummy binary labels

event_prediction_model = build_event_prediction_model(X_train.shape[1])

# Train the custom model
event_prediction_model.fit(X_train, y_train, epochs=5, batch_size=32)

# Step 4: Integrate the Models
def combine_models(frame):
    motion_results = detect_motion(frame)
    # Convert the motion results into a feature vector, if needed
    feature_vector = np.array([res['score'] for res in motion_results])
    event_prediction = event_prediction_model.predict(np.expand_dims(feature_vector, axis=0))
    return event_prediction

# Step 5: Deploy the Model Using Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_motion_event():
    frame = request.json['frame']  # Assume the frame is passed in JSON
    prediction = combine_models(frame)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    # Step 6: Real-Time Motion Detection with Streamlit
    st.title("Real-Time Motion Detection and Event Prediction")

    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display webcam feed on Streamlit
        st.image(frame, channels="BGR")

        # Detect motion using the Hugging Face model
        motion_detected = detect_motion(frame)
        st.write("Motion Detection Results:", motion_detected)

        # Run the event prediction model based on the motion detection
        event_prediction = combine_models(frame)
        st.write(f"Event Prediction: {event_prediction}")

        # Stop the video feed when 'Stop' button is clicked
        if st.button('Stop'):
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close OpenCV windows

    # Run Flask API in debug mode
    app.run(debug=True)
