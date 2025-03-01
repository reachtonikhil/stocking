import streamlit as st
import pandas as pd
import altair as alt
from nsepython import nse_eq
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Indian Stock Price Visualization and Analysis",
    page_icon="📈",
    layout="wide"
)

# Title of the application
st.title("📈 Indian Stock Price Visualization and Analysis")

# Sentiment analysis function using VADER
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']  # Returns a score between -1 (negative) and 1 (positive)

# Input: Stock symbol
stock_symbol = st.text_input("Enter the NSE stock symbol (e.g., RELIANCE):", value="RELIANCE")

# Fetch stock data
if stock_symbol:
    try:
        # Fetch the latest stock data
        stock_data = nse_eq(stock_symbol)
        
        # Extract relevant information
        company_name = stock_data['info']['companyName']
        last_price = stock_data['priceInfo']['lastPrice']
        change = stock_data['priceInfo']['change']
        p_change = stock_data['priceInfo']['pChange']
        day_high = stock_data['priceInfo']['intraDayHighLow']['max']
        day_low = stock_data['priceInfo']['intraDayHighLow']['min']
        previous_close = stock_data['priceInfo']['previousClose']
        
        # Display stock information
        st.subheader(f"{company_name} ({stock_symbol})")
        st.metric(label="Last Price", value=f"₹{last_price}", delta=f"{p_change}%")
        st.write(f"**Day High:** ₹{day_high}")
        st.write(f"**Day Low:** ₹{day_low}")
        st.write(f"**Previous Close:** ₹{previous_close}")
        st.write(f"**Change:** ₹{change} ({p_change}%)")
        
        # Prepare data for visualization
        price_data = {
            'Metric': ['Last Price', 'Day High', 'Day Low', 'Previous Close'],
            'Price': [last_price, day_high, day_low, previous_close]
        }
        df = pd.DataFrame(price_data)

        # Display bar chart for current price metrics
        st.markdown("### Current Price Metrics")
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Metric', sort=None),
            y='Price',
            color='Metric'
        ).properties(
            title=f"Price Metrics for {company_name}"
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # Fetch historical data using yfinance
        st.markdown("### Historical Stock Data")
        historical_data = yf.download(f"{stock_symbol}.NS", period="1y", interval="1d")
        historical_data.reset_index(inplace=True)
        st.write(historical_data.tail())

        # Calculate moving averages for additional features
        historical_data['MA_5'] = historical_data['Close'].rolling(window=5).mean()  # 5-day moving average
        historical_data['MA_20'] = historical_data['Close'].rolling(window=20).mean()  # 20-day moving average

        # Example News Headline for Sentiment Analysis
        news_headline = "TCS reports stellar quarterly earnings and optimistic future outlook."
        sentiment_score = get_sentiment(news_headline)
        st.write(f"Sentiment score for the news: {sentiment_score}")

        # Add sentiment score to the historical data
        historical_data['Sentiment'] = sentiment_score

        # Plot historical closing prices
        st.markdown("### Historical Closing Prices")
        line_chart = alt.Chart(historical_data).mark_line().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Close:Q', title='Closing Price (₹)'),
            tooltip=['Date:T', 'Close:Q']
        ).properties(
            title=f"Historical Closing Prices for {company_name}"
        )
        st.altair_chart(line_chart, use_container_width=True)

        # Prepare data for LSTM-based predictive analytics
        st.markdown("### Predictive Analytics with LSTM")
        historical_data['Day'] = np.arange(len(historical_data))
        features = historical_data[['Day', 'Sentiment', 'MA_5', 'MA_20']]  # Adding moving averages as features
        target = historical_data['Close']

        # Scale data for LSTM model
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(historical_data['Close'].values.reshape(-1, 1))

        # Prepare data for LSTM (using past 60 days to predict the next day)
        look_back = 60
        X_lstm, y_lstm = [], []
        for i in range(look_back, len(scaled_data)):
            X_lstm.append(scaled_data[i-look_back:i, 0])
            y_lstm.append(scaled_data[i, 0])

        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

        # Reshape data for LSTM input [samples, time steps, features]
        X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

        # Train-test split
        train_size = int(len(X_lstm) * 0.8)
        X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
        y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

        # Build LSTM model with dropout layers to prevent overfitting
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))  # Dropout layer for regularization
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))  # Dropout layer for regularization
        model.add(Dense(units=1))  # Output layer

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the LSTM model
        model.fit(X_train, y_train, epochs=20, batch_size=32)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions to get actual stock prices
        predicted_prices = scaler.inverse_transform(y_pred)

        # Format the predicted prices to two decimal places for proper rupees format
        formatted_predicted_prices = np.round(predicted_prices.flatten(), 2)

        # Display model evaluation metrics
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

        # Future prediction using LSTM
        st.markdown("### Future Price Prediction")
        future_days = st.slider("Select days for prediction:", 1, 30, 7)

        # Get the last day from the historical data
        last_day = historical_data['Day'].iloc[-1]

        # Generate future day indices relative to the last day
        future_day_indices = np.arange(last_day + 1, last_day + 1 + future_days)

        # Ensure predictions are flattened to 1D
        future_predictions = model.predict(future_day_indices.reshape(-1, 1)).flatten()

        # Inverse scaling for future predictions
        future_predictions_scaled = scaler.inverse_transform(future_predictions.reshape(-1, 1))

        # Format future predicted prices to 2 decimal places
        formatted_future_predictions = np.round(future_predictions_scaled.flatten(), 2)

        # Create a DataFrame for future predictions with formatted values
        future_df = pd.DataFrame({
            'Predicted Price (₹)': formatted_future_predictions  # Remove the Day column
        })

        # Add the index as a column for Altair to use it in the plot
        future_df['Day Index'] = future_df.index + 1  # Adding 1 to start from 1 instead of 0

        # Display the DataFrame with proper formatting (remove 'Day' column)
        st.write(future_df)

        # Display future price predictions using Altair with formatted prices
        prediction_chart = alt.Chart(future_df).mark_line(color='red').encode(
            x=alt.X('Day Index:O', title='Day Index'),  # Use 'Day Index' for the x-axis
            y=alt.Y('Predicted Price (₹):Q', title='Predicted Price (₹)')
        ).properties(
            title=f"Future Price Predictions for {company_name}"
        )
        st.altair_chart(prediction_chart, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
