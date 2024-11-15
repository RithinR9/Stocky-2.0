import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'DIS', 'JPM', 
          'BAC', 'V', 'MA', 'HD', 'PG', 'UNH', 'VZ', 'INTC', 'CSCO', 'KO', 'PEP', 'MRK']

def get_user_input():
    budget = float(input("Enter your investment budget: "))
    time_frame = int(input("Enter investment time frame in years: "))
    return budget, time_frame

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")  # Fetch last 5 years of data
    return hist[['Close']]

def load_model(stock_data, stock_name):
    """
    Trains a linear regression model if not already saved, otherwise loads it from file.
    :param stock_data: A dataframe with 'Close' prices.
    :param stock_name: The stock ticker symbol.
    :return: Trained linear regression model.
    """
    model_path = "model.joblib"
    # Load the model if it exists
    model = joblib.load(model_path)    
        
    
    return model

def predict_future_price(stock_data, model, days_ahead):
    """
    Predicts future stock price using a pre-trained linear regression model.
    :param stock_data: Dataframe with 'Close' prices.
    :param model: Pre-trained linear regression model.
    :param days_ahead: Days into the future to predict.
    :return: Predicted price.
    """
    future_day = [[len(stock_data) + days_ahead]]
    future_price = model.predict(future_day)
    return future_price[0]

def recommend_stock(budget, time_frame):
    recommendations = []
    
    for stock in STOCKS:
        hist = fetch_stock_data(stock)
        if hist.empty:
            continue  # Skip if no data available
        
        # Train or load a model for the current stock
        model = load_model(hist, stock)
        
        # Predict future price at the end of the investment time frame
        days_ahead = time_frame * 252  # Approx. trading days per year
        predicted_price = predict_future_price(hist, model, days_ahead)
        current_price = hist['Close'][-1]

        # Only recommend if within budget and predicted growth
        if current_price <= budget and predicted_price > current_price:
            recommendations.append({
                "Stock": stock,
                "Current Price": current_price,
                "Predicted Price": predicted_price,
                "Expected Growth (%)": ((predicted_price - current_price) / current_price) * 100
            })
    
    # Sort recommendations by expected growth in descending order
    recommendations = sorted(recommendations, key=lambda x: x['Expected Growth (%)'], reverse=True)
    return recommendations

def display_recommendations(recommendations):
    if not recommendations:
        print("No suitable stocks found within your budget and time frame.")
    else:
        print("Recommended Stocks:")
        for rec in recommendations:
            print(f"Stock: {rec['Stock']}, Current Price: ${rec['Current Price']:.2f}, "
                  f"Predicted Price in {time_frame} years: ${rec['Predicted Price']:.2f}, "
                  f"Expected Growth: {rec['Expected Growth (%)']:.2f}%")

# Main Program
budget, time_frame = get_user_input()
recommendations = recommend_stock(budget, time_frame)
display_recommendations(recommendations)
