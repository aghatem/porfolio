
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Alpha Vantage API Configuration ---
# You need to get your API key from Alpha Vantage: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
ALPHA_VANTAGE_API_KEY = "GERY9UON8HH4FWK2" # <<< IMPORTANT: Replace with your actual API key

# --- Constants ---
TODAY = date.today()
START = TODAY - timedelta(days=365 * 5) # 5 years of data

# --- Helper Functions for Alpha Vantage ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_alpha_vantage_data(symbol, interval="daily", outputsize="full"):
    """
    Fetches historical stock data from Alpha Vantage.

    Args:
        symbol (str): The stock ticker symbol (e.g., "IBM").
        interval (str): The data interval ("daily", "weekly", "monthly").
                        For daily, use "TIME_SERIES_DAILY_ADJUSTED".
        outputsize (str): "compact" returns only the latest 100 data points,
                          "full" returns the full-length time series.

    Returns:
        pd.DataFrame: A DataFrame with historical stock data, or None if an error occurs.
    """
    if interval == "daily":
        function = "TIME_SERIES_DAILY_ADJUSTED"
    elif interval == "weekly":
        function = "TIME_SERIES_WEEKLY_ADJUSTED"
    elif interval == "monthly":
        function = "TIME_SERIES_MONTHLY_ADJUSTED"
    else:
        st.error(f"Unsupported interval: {interval}. Please choose 'daily', 'weekly', or 'monthly'.")
        return None

    url = f"[https://www.alphavantage.co/query?function=](https://www.alphavantage.co/query?function=){function}&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if "Error Message" in data:
            st.error(f"Alpha Vantage API Error for {symbol}: {data['Error Message']}")
            return None
        if "Note" in data:
            st.warning(f"Alpha Vantage API Note for {symbol}: {data['Note']}")

        time_series_key = f"Time Series ({interval.capitalize()} Adjusted)" if interval == "daily" else f"{interval.capitalize()} Adjusted Time Series"

        if time_series_key not in data:
            st.error(f"Could not find '{time_series_key}' in Alpha Vantage response for {symbol}.")
            return None

        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume"
        })
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df = df.sort_index() # Sort by date ascending

        # Filter by date range as Alpha Vantage doesn't directly support it for daily
        df = df[(df.index >= pd.to_datetime(START)) & (df.index <= pd.to_datetime(TODAY))]
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return None
    except ValueError as e:
        st.error(f"Error parsing Alpha Vantage JSON response: {e}")
        return None

def get_stock_info_alpha_vantage(symbol):
    """
    Fetches basic stock information (like company name) from Alpha Vantage.
    Using GLOBAL_QUOTE for a quick check, but OVERVIEW gives more details.
    """
    url = f"[https://www.alphavantage.co/query?function=OVERVIEW&symbol=](https://www.alphavantage.co/query?function=OVERVIEW&symbol=){symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "Name" in data:
            return data["Name"]
        elif "Error Message" in data:
            st.error(f"Alpha Vantage API Error for {symbol}: {data['Error Message']}")
            return None
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching stock info from Alpha Vantage: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Stock Price Predictor")

st.title("📈 Stock Price Predictor")
st.write("Enter a stock ticker symbol to view its historical prices and predict the next day's closing price using an LSTM model.")

# --- Stock Selection ---
stock_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "V", "PG"] # Example stocks
selected_stock = st.sidebar.selectbox("Select a stock or enter a custom one:", stock_options + ["Custom"])

custom_stock_ticker = ""
if selected_stock == "Custom":
    custom_stock_ticker = st.sidebar.text_input("Enter custom stock ticker (e.g., IBM):").upper()
    ticker_symbol = custom_stock_ticker
else:
    ticker_symbol = selected_stock

# Interval selection
selected_interval = st.sidebar.selectbox("Select data interval:", ["daily", "weekly", "monthly"])

if ticker_symbol:
    # --- Fetch Stock Information (Avant API equivalent) ---
    st.subheader(f"Analyzing: {ticker_symbol}")
    company_name = get_stock_info_alpha_vantage(ticker_symbol)
    if company_name:
        st.write(f"Company Name: **{company_name}**")
    else:
        st.write("Could not retrieve company name.")

    # --- Fetch Historical Data (Avant API equivalent) ---
    # Line 124 (original): data = yf.download(ticker, START, TODAY, interval=selected_interval)
    data = get_alpha_vantage_data(ticker_symbol, interval=selected_interval)

    if data is not None and not data.empty:
        st.subheader("Historical Data")
        st.dataframe(data.tail()) # Show last few rows of data

        # Plotting Historical Data
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.update_layout(title=f'{ticker_symbol} Historical Price',
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Prepare Data for LSTM ---
        st.subheader("Building LSTM Model...")
        # Use 'Close' price for prediction
        data_close = data['Close'].values.reshape(-1, 1)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_close)

        # Create training data set
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_data_len, :]

        # Prepare x_train and y_train
        x_train = []
        y_train = []
        look_back = 60 # Number of previous days to consider for prediction

        for i in range(look_back, len(train_data)):
            x_train.append(train_data[i-look_back:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model (can take a while for large datasets)
        with st.spinner("Training the LSTM model... This might take a few moments."):
            model.fit(x_train, y_train, batch_size=1, epochs=1) # Reduced epochs for faster demo

        # Create the testing data set
        test_data = scaled_data[training_data_len - look_back:, :]
        x_test = []
        y_test = data_close[training_data_len:, :]

        for i in range(look_back, len(test_data)):
            x_test.append(test_data[i-look_back:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get the model's predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        st.subheader("Model Predictions vs. Actual Prices")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train Price'))
        fig2.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual Price'))
        fig2.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))
        fig2.update_layout(title=f'Model Predictions for {ticker_symbol}',
                           xaxis_title='Date',
                           yaxis_title='Close Price',
                           legend_title='Legend')
        st.plotly_chart(fig2, use_container_width=True)

        # --- Predict Next Day's Price ---
        st.subheader("Next Day Price Prediction")
        if not data.empty:
            # Get the last 'look_back' days of data
            last_look_back_days = data_close[-look_back:]
            last_look_back_days_scaled = scaler.transform(last_look_back_days)

            # Reshape for prediction
            X_next_day = np.array([last_look_back_days_scaled])
            X_next_day = np.reshape(X_next_day, (X_next_day.shape[0], X_next_day.shape[1], 1))

            # Predict the next day's price
            next_day_prediction_scaled = model.predict(X_next_day)
            next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)

            st.success(f"The predicted closing price for **{ticker_symbol}** for the next trading day is: **${next_day_prediction[0][0]:.2f}**")
        else:
            st.warning("Not enough data to predict the next day's price.")

    elif data is None:
        st.error(f"Failed to retrieve data for {ticker_symbol}. Please check the ticker symbol and your API key.")
    else:
        st.warning(f"No data available for {ticker_symbol} in the specified date range ({START} to {TODAY}).")
else:
    st.info("Please select or enter a stock ticker symbol to begin.")

