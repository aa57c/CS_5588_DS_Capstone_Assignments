








import yfinance as yf
import praw
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')

# Stock data from yfinance
stock = yf.Ticker("AAPL")
stock_data = stock.history(period='1y')

# Reddit data using praw
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
subreddit = reddit.subreddit('StockPrice')
comments = [comment.body for comment in subreddit.comments(limit=10)]

print("==== Comments Gathered ====")
print(comments)

from transformers import pipeline
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiments = [sentiment_model(comment)[0]['label'] for comment in comments]
print("Reddit Comment Sentiments:", sentiments)



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare stock data
data = stock_data[['Close']].values

X, y = [], []
sequence_length = 10

# Create sequences of 10 time steps
for i in range(sequence_length, len(data)):
    X.append(data[i-sequence_length:i])
    y.append(data[i])

# Convert X and y to numpy arrays
X, y = np.array(X), np.array(y)

# Debug: Print shapes of X and y
print(f"Shape of X before reshaping: {X.shape}")  # Should be (samples, time steps)
print(f"Shape of y: {y.shape}")  # Should be (samples,)

# Reshape X to fit the LSTM input: (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Debug: Print the shape of X after reshaping
print(f"Shape of X after reshaping: {X.shape}")  # Should be (samples, time steps, 1)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),  # Time steps, Features
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=5, batch_size=32)


def combine_models(stock_prediction, reddit_sentiment):
    adjustment = 0.05 * stock_prediction if 'POSITIVE' in reddit_sentiment else -0.05 * stock_prediction
    return stock_prediction + adjustment

final_predictions = [combine_models(pred, sent) for pred, sent in zip(stock_data['Close'][-len(sentiments):], sentiments)]

import streamlit as st

st.title("Real-Time Stock and Reddit Sentiment Prediction")

# Displaying real-time data and predictions
st.line_chart(stock_data['Close'])
st.write("Reddit Sentiments:", sentiments)
st.write("Final Stock Prediction:", final_predictions)







