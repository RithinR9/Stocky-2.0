import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import font
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib as mpl
import yfinance as yf

mpl.rcParams['font.size'] = 9

# Create the main window
window = tk.Tk()
window.title("STOCKY-AI STOCK BOT BY RITHIN")

# Set dark theme
window.configure(bg='#1E1E1E')
window.tk_setPalette(background='#1E1E1E', foreground='#FFFFFF')

roboto_font = font.Font(family='Roboto Mono', size=12)

# Create a label and input box for the stock symbol
symbol_label = tk.Label(window, text="Please enter a stock symbol (e.g. AAPL): ", font=(roboto_font))
symbol_label.pack()
entry = tk.Entry(window, font=(roboto_font))
entry.pack()

# Create a button to submit the stock symbol
submit_button = tk.Button(window, text="Submit", command=None, font=(roboto_font))
submit_button.pack()

# Create a label
result_label = tk.Label(window, font=(roboto_font))
result_label.pack()

# Create a canvas outside of the predict function
canvas = None

def update_canvas(df, csv, predictions):
    global canvas  # Use the global canvas
    if canvas:
        canvas.get_tk_widget().destroy()  # Clear the previous canvas

    fig = Figure(figsize=(12, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(df['Date'], df['Close'], label='Historical Close Price')

    # Plot the 30-day predictions
    if predictions is not None:
        prediction_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
        ax.plot(prediction_dates, predictions, 'ro-', label='Predicted Close Price')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)
    ax.set_title(f'{csv} Stock Price Prediction (30 Days)', fontsize=12)
    ax.legend(fontsize=12)

    fig.autofmt_xdate(rotation=45)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

def predict():
    global canvas  # Use the global canvas

    # Get user input
    csv = entry.get().upper()

    # Fetch stock data
    df = yf.Ticker(csv).history(period="3mo").reset_index()
    if df.empty:
        result_label.config(text="Invalid stock symbol")
        return

    # Prepare data
    df['Daily % Change'] = (df['Close'].pct_change() * 100).round(2)
    df['Next Day % Change'] = df['Daily % Change'].shift(-1)
    df = df.dropna().reset_index(drop=True)

    # Define feature and target variables
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Next Day % Change']
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Start with the last row data and predict 30 days into the future
    last_row = X.iloc[-1].copy()
    predictions = []

    for i in range(30):
        # Predict the percentage change
        predicted_change = reg.predict([last_row])[0]
        
        # Calculate the predicted close price
        last_close_price = df['Close'].iloc[-1] if i == 0 else predictions[-1]
        predicted_close_price = last_close_price * (1 + predicted_change / 100)
        predictions.append(predicted_close_price)

        # Update `last_row` with the predicted data for next iteration, adding slight variation
        last_row['Open'] = predicted_close_price * (1 + np.random.normal(0, 0.01))  # Small random variation
        last_row['High'] = predicted_close_price * (1 + np.random.normal(0.02, 0.02))
        last_row['Low'] = predicted_close_price * (1 - np.random.normal(0.02, 0.02))
        last_row['Close'] = predicted_close_price
        last_row['Volume'] = last_row['Volume'] * (1 + np.random.normal(0, 0.05))  # Small variation in volume

        # Display the predicted percentage change in stock prices for the next day
        result_label.config(
            text=(
                f"Predicted percentage change for the next day: {predicted_change:.2f}%\n"
                f"Change: {10000 * predicted_change / 100:.2f}\n"
                f"Potential Profit: {10000 * (1 + predicted_change / 100):.2f}\n"
                f"Chance of profit: {'Good' if predicted_change > 0 else 'Bad'}"
            )
        )

    # Update the canvas with the 30-day predictions
    update_canvas(df, csv, predictions=predictions)

# Set the command for the submit button
submit_button.config(command=predict)

window.wm_state('zoomed')
window.mainloop()
