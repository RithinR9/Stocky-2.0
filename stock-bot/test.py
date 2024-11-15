import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import font
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
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
window.configure(bg='#1E1E1E')  # Background color
window.tk_setPalette(background='#1E1E1E', foreground='#FFFFFF')  # Text color


roboto_font = font.Font(family='Roboto Mono', size=12)

# Create a label and input box for the stock symbol
symbol_label = tk.Label(window, text="Please enter a stock symbol (e.g. AAPL): ", font=(roboto_font))
symbol_label.pack()
entry = tk.Entry(window, font=(roboto_font))
entry.pack()

# Create a button to submit the stock symbol
submit_button = tk.Button(window, text="Submit", command=None, font=(roboto_font))  # Initialize with None
submit_button.pack()

# Create a label
result_label = tk.Label(window, font=(roboto_font))
result_label.pack()

# Create a canvas outside of the predict function
canvas = None

def update_canvas(df, csv, predicted_change=None):
    global canvas  # Use the global canvas
    if canvas:
        canvas.get_tk_widget().destroy()  # Clear the previous canvas

    date = datetime.now()
    # Create a figure and axis for the graph
    fig = Figure(figsize=(12, 4), dpi=100)
    ax = fig.add_subplot(111)

    # Plot the previous stock prices
    ax.plot(df['Date'], df['Close'])

    # Add a title and labels to the graph
    ax.set_title('Stock Prices', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)

    # Plot a line graph of the Close price for the past 42 days and the predicted Close price for the next day
    fig = Figure(figsize=(5, 4), dpi=100)
    plot = fig.add_subplot(1, 1, 1)
    plot.plot(df['Date'], df['Close'], label='Historical Close Price')
    
    # Check if predicted_change is provided and plot it
    if predicted_change is not None:
        next_day_index = len(df)
        predicted_close_price = df['Close'].iloc[-1] * (1 + predicted_change / 100)
        plot.plot(df['Date'].iloc[next_day_index - 1], predicted_close_price, 'ro', label='Tomorrow\'s Predicted Close Price')

    plot.set_xlabel('Date', fontsize=12)
    plot.set_ylabel('Close Price', fontsize=12)
    plot.set_title(f'{csv} Stock Price ({date})', fontsize=12)
    plot.legend(fontsize=12)

    fig.autofmt_xdate(rotation=45)

    # Create a Tkinter canvas to display the graph
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


def predict():
    global canvas  # Use the global canvas
    # Calculate the time range for the API URL (last 42 days)

    # Get user input
    csv = entry.get()

    # Construct the URL to retrieve the CSV file
    url = yf.Ticker(csv.capitalize()).history(period="1mo").to_csv('stock.csv')

    # Retrieve the CSV file
    if 1 == 2:
        # Display an error message if the symbol is not valid
        result_label.config(text="Invalid stock symbol")
    else:
        for i in range(1, 31):
            # Load the CSV file into a Pandas DataFrame
            df = pd.read_csv('stock.csv')
            predicted_change = 0

            # Add a column to the DataFrame that predicts the percentage change in stock prices from one day to the next
            df['Daily % Change'] = (df['Close'].pct_change() * 100).round(2)

            df['Next Day % Change'] = df['Daily % Change'].shift(-1)

            # Drop the last row since it doesn't have a prediction target
            df.drop(df.tail(1).index, inplace=True)

            # Split the data into training and testing sets
            X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = df['Next Day % Change']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

            # Train a linear regression model
            reg = LinearRegression()
            reg.fit(X_train, y_train)

            # Make predictions on the testing set
            y_pred = reg.predict(X_test)

            # Calculate the predicted percentage change, as well as the corresponding change and potential profit for the next day based on a hypothetical investment of $10,000
            predicted_change = y_pred[-1]
            
            investment = 10000
            change = investment * predicted_change / 100
            profit = investment + change

            # Calculate the chance of profit
            if predicted_change > 0:
                chance_of_profit = "Good"
            else:
                chance_of_profit = "Bad"

            # Create a DataFrame with the prediction data
            prediction_data = {
                'Date': [pd.Timestamp.now() + pd.Timedelta(days=i)],
                'Open': [float(df['Open'].mean())],  # Placeholder for Open
                'High': [float(df['High'].mean())],  # Placeholder for High
                'Low': [float(df['Low'].mean())],   # Placeholder for Low
                'Close': [predicted_change], # Replace Close with Predicted Change
                'Volume': [float(df['Volume'].mean())],# Placeholder for Volume
                'Dividends': [0.0],
                'Stock Splits': [float(df['Stock Splits'].mean())],
            }
            prediction_df = pd.DataFrame(prediction_data)

            # Load the existing stock data
            stock_df = pd.read_csv('stock.csv')

            # Append the prediction data to the existing stock data
            stock_df = pd.concat([stock_df, prediction_df], ignore_index=True)

            # Save the updated DataFrame back to the CSV file
            stock_df.to_csv('stock.csv', index=False)

            # Display the predicted percentage change in stock prices for the next day as a percentage
            result_label.config(
                text=
                f"Predicted percentage change for the next day: {predicted_change:.2f}%\nChange: {change:.2f}\nPotential Profit: {profit:.2f}\nChance of profit: {chance_of_profit}"
            )

        # Update the canvas
        # Update the canvas with the predicted change
        update_canvas(df, csv, predicted_change=predicted_change)


# Set the command for the submit button 
submit_button.config(command=predict)

window.wm_state('zoomed')

window.mainloop()
