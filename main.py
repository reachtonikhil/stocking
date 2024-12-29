import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from plotly import graph_objs as go

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Simplified Stock Analysis",
    page_icon="📈",
    layout="wide"
)

# Title of the application
st.title("📈 Stock Price Analysis and Basic Prediction")

# Input: Stock symbol
stock_symbol = st.text_input("Enter the stock symbol (e.g., RELIANCE):", value="RELIANCE")

if stock_symbol:
    try:
        # Fetch historical stock data using yfinance
        stock = yf.Ticker(stock_symbol + ".NS")
        historical_data = stock.history(period="1y")
        historical_data.reset_index(inplace=True)

        # Display basic stock information
        st.subheader(f"📊 Historical Data for {stock_symbol}")
        st.write(historical_data[['Date', 'Close']].tail(10))

        # Candlestick chart using Plotly
        st.subheader("📊 Historical Price Data (Candlestick Chart)")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=historical_data['Date'],
            open=historical_data['Open'],
            high=historical_data['High'],
            low=historical_data['Low'],
            close=historical_data['Close'],
            name="Candlestick"
        ))
        fig.update_layout(title=f"1-Year Price Data for {stock_symbol}", xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

        # Prepare data for basic prediction
        st.subheader("🔮 Price Prediction for the Next Month")
        historical_data['Day'] = np.arange(len(historical_data))  # Add a numeric day column
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(historical_data[['Close']])
        days = historical_data['Day'].values.reshape(-1, 1)

        # Train a simple Linear Regression model
        model = LinearRegression()
        model.fit(days, scaled_close)

        # Predict the next 30 days
        future_days = np.arange(len(historical_data), len(historical_data) + 30).reshape(-1, 1)
        predictions = model.predict(future_days)
        predicted_prices = scaler.inverse_transform(predictions)

        # Display predicted prices
        future_dates = [historical_data['Date'].max() + timedelta(days=i) for i in range(1, 31)]
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices.flatten()})
        st.write(prediction_df)

        # Line chart for predicted prices
        st.subheader("📈 Predicted Prices for the Next 30 Days")
        st.line_chart(prediction_df.set_index("Date"), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
