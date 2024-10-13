############################################
#   ___        _   _   _   _   _           #
#  |_ _|_ __  (_) (_) (_) (_) (_)   ___    #
#   | || '_ \ | | | | | | | | | |  / _ \   #
#   | || | | || | | | | | | | | | |  __/   #
#  |___|_| |_||_| |_| |_| |_| |_|  \___|   #
#                                          #
#    REAL-TIME PREDICTION SYSTEM           #
#        USING RANDOM FOREST MODEL         #
############################################

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
import time

def fetch_live_data(ticker):
    """
    Fetch live cryptocurrency data for BTC.
    """
    live_data = yf.download(ticker, period='1d', interval='1m')
    return live_data

# ================================================================
# Handle missing values and extract relevant features
# ================================================================
def preprocess_data(data):
    """
    Preprocess the data: Handle missing values and select features.
    """
    data.ffill(inplace=True)  # Fill missing values
    features = data[['Open', 'High', 'Low', 'Volume']]
    return features

# ================================================================
# Model->[(Random Forest)]
# ================================================================
def train_model(X, y):
    """
    Train a Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)  # Fit the model to the training data
    return model

# ================================================================
#  [Further Training]
# ================================================================
def save_model(model, filename):
    """
    Save the trained model using joblib.
    """
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

# ================================================================
#  [Live Data Stream]
# ================================================================
def predict_live_price(model, live_data):
    """
    Make a live prediction using the trained model and live data.
    """
    features = preprocess_data(live_data)
    prediction = model.predict(features)
    current_price = live_data['Close'].iloc[-1]  # Latest actual price

    predicted_price = prediction[-1]  # Latest predicted price
    percent_error = abs((predicted_price - current_price) / current_price) * 100  # Percent error

    return predicted_price, current_price, percent_error

# ================================================================
#  [Graph Trend]
# ================================================================
def display_trend(current_price, predicted_price):
    """
    Display a simple ASCII graph showing the trend direction.
    """
    diff = predicted_price - current_price
    if diff > 0:
        trend = "Upwards"
        graph = "/" * min(10, int(diff // 100))  # Scale for the graph (limit to 10 characters)
    else:
        trend = "Downwards"
        graph = "\\" * min(10, abs(int(diff // 100)))  # Scale for the graph (limit to 10 characters)

    print(f"Trend: {trend} {graph}")

# ================================================================
#  [Check for Pickle before running]
# ================================================================
if __name__ == '__main__':
    # Model file path
    model_filename = 'btc_price_model.pkl'

    # Check if a pre-trained model exists
    if os.path.exists(model_filename):
        print(f"Loading existing model from {model_filename}...")
        model = joblib.load(model_filename)  # Load the pre-trained model
    else:
        print("Training a new model...")
        # Fetch historical stock data (last 1 month, hourly interval)
        stock_data = yf.download('BTC-USD', period='1mo', interval='1h')

        # Data preprocessing: Features and target (Close price)
        X = preprocess_data(stock_data)
        y = stock_data['Close']

        # Split the data into training and test sets (80% training, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Random Forest model
        model = train_model(X_train, y_train)

        # Evaluate the model using Mean Squared Error (MSE)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE): {mse}")

        # Save the trained model
        save_model(model, model_filename)

    # ------------------------------------------------------------
    # Real-time prediction loop: Fetch live data and predict
    # ------------------------------------------------------------
    prediction_count = 0
    total_predictions = 100  # Set the number of predictions before stopping

    while prediction_count < total_predictions:
        live_data = fetch_live_data('BTC-USD')
        predicted_price, current_price, percent_error = predict_live_price(model, live_data)
        
        # Increment prediction counter
        prediction_count += 1

        # Display the results
        print(f"\nPrediction {prediction_count} of {total_predictions} completed:")
        print(f"Predicted BTC Price: {predicted_price:.2f}")
        print(f"Actual BTC Price: {current_price:.2f}")
        print(f"Percent Error: {percent_error:.2f}%")

        # trend graph
        display_trend(current_price, predicted_price)

        # Progress bar: Shows the progress of how many predictions have been done
        progress_percentage = int((prediction_count / total_predictions) * 100)
        print(f"[{'*' * progress_percentage}{' ' * (100 - progress_percentage)}]  {prediction_count} of {total_predictions} completed.")

        # Wait 1 minute before fetching the next set of data
        time.sleep(60)

    print("All predictions completed.")
