import streamlit as st
import pandas as pd
import altair as alt
from nsepython import nse_eq
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Indian Stock Price Visualization",
    page_icon="📈",
    layout="wide"
)

# Title of the application
st.title("📈 Indian Stock Price Visualization and Analysis")

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
        historical_data = yf.download(f"{stock_symbol}.NS", period="6mo", interval="1d")
        historical_data.reset_index(inplace=True)
        st.write(historical_data.tail())

        # Plot historical closing prices
        st.markdown("### Historical Closing Prices")
        line_chart = alt.Chart(historical_data).mark_line().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Close', title='Closing Price (₹)'),
            tooltip=['Date:T', 'Close']
        ).properties(
            title=f"Historical Closing Prices for {company_name}"
        )
        st.altair_chart(line_chart, use_container_width=True)

        # Calculate and display moving averages
        st.markdown("### Moving Averages")
        historical_data['SMA_10'] = historical_data['Close'].rolling(window=10).mean()
        historical_data['SMA_30'] = historical_data['Close'].rolling(window=30).mean()
        moving_avg_chart = alt.Chart(historical_data).mark_line().encode(
            x='Date:T',
            y=alt.Y('Close', title='Price (₹)'),
            color=alt.value('blue')
        ).properties(
            title=f"Moving Averages for {company_name}"
        ) + alt.Chart(historical_data).mark_line(color='orange').encode(
            x='Date:T',
            y=alt.Y('SMA_10', title='SMA 10 Days')
        ) + alt.Chart(historical_data).mark_line(color='green').encode(
            x='Date:T',
            y=alt.Y('SMA_30', title='SMA 30 Days')
        )
        st.altair_chart(moving_avg_chart, use_container_width=True)

        # Predictive Analytics
        st.markdown("### Predictive Analytics")
        historical_data['Day'] = np.arange(len(historical_data))
        X = historical_data[['Day']]
        y = historical_data['Close']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Display metrics
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

        # Future prediction
        st.markdown("### Future Price Prediction")
        future_days = st.slider("Select days for prediction:", 1, 30, 7)
        future_day_indices = np.arange(len(historical_data), len(historical_data) + future_days).reshape(-1, 1)
        future_predictions = model.predict(future_day_indices)

        future_df = pd.DataFrame({
            'Day': np.arange(len(historical_data), len(historical_data) + future_days),
            'Predicted Price': future_predictions
        })
        st.write(future_df)

        # Display future price predictions
        prediction_chart = alt.Chart(future_df).mark_line(color='red').encode(
            x=alt.X('Day', title='Day Index'),
            y=alt.Y('Predicted Price', title='Predicted Price (₹)')
        ).properties(
            title=f"Future Price Predictions for {company_name}"
        )
        st.altair_chart(prediction_chart, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
