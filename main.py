import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from datetime import datetime, timedelta
from plotly import graph_objs as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from nsepython import nse_eq
from transformers import pipeline

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="AI-Driven Stock Analysis",
    page_icon="📈",
    layout="wide"
)

# Title of the application
st.title("📈 AI-Driven Stock Analysis and Trend Prediction")

# Input: Stock symbol
stock_symbol = st.text_input("Enter the NSE stock symbol (e.g., RELIANCE):", value="RELIANCE")

# Sentiment analysis pipeline
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

sentiment_model = load_sentiment_model()

if stock_symbol:
    try:
        # Fetch live stock data using nsepython
        live_data = nse_eq(stock_symbol)
        company_name = live_data['info']['companyName']
        last_price = live_data['priceInfo']['lastPrice']
        change = live_data['priceInfo']['change']
        p_change = live_data['priceInfo']['pChange']
        day_high = live_data['priceInfo']['intraDayHighLow']['max']
        day_low = live_data['priceInfo']['intraDayHighLow']['min']
        previous_close = live_data['priceInfo']['previousClose']

        # Display live stock information
        st.subheader(f"{company_name} ({stock_symbol})")
        st.metric(label="Last Price", value=f"₹{last_price}", delta=f"{p_change}%")
        st.write(f"**Day High:** ₹{day_high}")
        st.write(f"**Day Low:** ₹{day_low}")
        st.write(f"**Previous Close:** ₹{previous_close}")
        st.write(f"**Change:** ₹{change} ({p_change}%)")

        # Fetch historical stock data using yfinance
        stock = yf.Ticker(stock_symbol + ".NS")
        historical_data = stock.history(period="1y")
        historical_data.reset_index(inplace=True)

        # Date picker for specific date selection
        st.subheader("📅 Select a Date for Trend Prediction")
        selected_date = st.date_input("Choose a date from historical data:", min_value=historical_data['Date'].min(), max_value=historical_data['Date'].max())
        if selected_date:
            selected_data = historical_data[historical_data['Date'] == pd.Timestamp(selected_date)]
            if not selected_data.empty:
                st.write(f"**Selected Date's Close Price:** ₹{selected_data['Close'].values[0]:.2f}")
            else:
                st.error("Selected date not available in the data.")

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
        fig.update_layout(title=f"1-Year Price Data for {company_name}", xaxis_title="Date", yaxis_title="Price (₹)")
        st.plotly_chart(fig, use_container_width=True)

        # LSTM-based stock price trend prediction
        st.subheader("🔮 Stock Price Trend Prediction for the Next Month")
        scaler = MinMaxScaler(feature_range=(0, 1))
        historical_close = historical_data[['Close']]
        scaled_data = scaler.fit_transform(historical_close)

        # Prepare data for modeling
        def create_dataset(data, time_step=60):
            x_data, y_data = [], []
            for i in range(len(data) - time_step - 1):
                x_data.append(data[i:(i + time_step), 0])
                y_data.append(data[i + time_step, 0])
            return np.array(x_data), np.array(y_data)

        time_step = 60
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Define LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        # Predict next 30 days
        recent_data = scaled_data[-time_step:]
        predictions = []
        input_seq = recent_data.reshape(1, time_step, 1)
        for _ in range(30):
            predicted = model.predict(input_seq)
            predictions.append(predicted[0, 0])
            input_seq = np.append(input_seq[:, 1:, :], [[predicted]], axis=1)

        # Scale predictions back
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Display predicted prices as a line chart
        future_dates = [historical_data['Date'].max() + timedelta(days=i) for i in range(1, 31)]
        prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predicted_prices.flatten()})
        st.line_chart(prediction_df.set_index("Date"), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
