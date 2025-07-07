import streamlit as st
from datetime import date
import requests
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import seaborn as sns
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

# --- Alpha Vantage API Key ---
# It's recommended to store this as a Streamlit secret, but we'll define it here for simplicity.
ALPHA_VANTAGE_API_KEY = 'GERY9UON8HH4FWK2'

# --- Constants for Web Scraping ---
GAINERS_URL = 'https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/'
LOSERS_URL = 'https://www.tradingview.com/markets/stocks-usa/market-movers-losers/'

# --- Functions ---

@st.cache_data(ttl=3600) # Cache the data for 1 hour
def get_top_movers_list(url):
    """
    Scrapes the top movers (gainers or losers) from TradingView.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        rows = table.find_all("tr")
        movers = []
        for row in rows[1:]:
            cells = row.find_all("td")
            symbol_comp = cells[0]
            result = str(symbol_comp).split('title="', 1)[-1].strip()
            result = result.split('">', 1)[0].strip()
            symbol, company_name = result.split(' − ', 1)
            change_percent = cells[1].text.strip()
            price = cells[2].text.strip()
            change_percentage = cells[3].text.strip()
            movers.append({
                "Symbol": symbol,
                "Company Name": company_name,
                "Change_perent": change_percent,
                "Price": price,
                "Change % 1D": change_percentage
            })
        return pd.DataFrame(movers)
    except Exception as e:
        st.error(f"Could not fetch top movers: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_data(ticker, interval):
    """
    Loads stock data from Alpha Vantage for a given ticker and interval.
    """
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data = None
        
        if interval == '1d':
            # Use get_daily_adjusted to include 'Adj Close'
            data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        elif interval == '1wk':
            data, _ = ts.get_weekly_adjusted(symbol=ticker)
        # Note: Alpha Vantage free tier has limitations on intraday and monthly calls.
        # Add more interval options as needed.
        else:
            st.error(f"Interval '{interval}' is not supported by this version of the app.")
            return pd.DataFrame()

        if data is None or data.empty:
             st.warning(f"No data returned from Alpha Vantage for ticker {ticker}.")
             return pd.DataFrame()

        # Rename columns to a standard format
        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume'
        }, inplace=True)

        # Convert index to datetime and sort
        data.index = pd.to_datetime(data.index)
        data.sort_index(ascending=True, inplace=True)
        return data

    except Exception as e:
        st.error(f"An error occurred while fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_company_info(ticker):
    """
    Fetches company overview data from Alpha Vantage.
    """
    try:
        fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        info, _ = fd.get_company_overview(symbol=ticker)
        return info.iloc[0] # Return the first row as a Series
    except Exception as e:
        st.error(f"Could not fetch company info for {ticker}: {e}")
        return None

# --- Streamlit App Layout ---

st.title('Stock Forecast App')

st.write('This web app will provide you the information needed to analyze the stock market and will rely on a machine learning model to predict the estimated value.')
st.write('It will rely on the historical stock value from the **Alpha Vantage API** (in addition to other information like Market Cap and details on the company). ')
st.write('This web app will help you to analyze the daily top 100 tickers gainers and loosers as it performs HTML parsing from tradingview.com to provide you with the daily lists.')
st.write('Eventually the historical prices will be fed into a Machine Learning Tensorflow model to be trained and then predict the closing value of the selected stock.')

# --- Data Loading ---
Gainers_list = get_top_movers_list(GAINERS_URL)
Loosers_list = get_top_movers_list(LOSERS_URL)

with st.expander("Click to see today's top movers"):
    list_choice = ('Gainers', 'Loosers')
    selected_movers = st.selectbox('Select the list:', list_choice)
    if selected_movers == 'Gainers':
        st.dataframe(Gainers_list)
    else:
        st.dataframe(Loosers_list)

# --- User Inputs ---
stocks = {'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'META', 'TSLA', 'NVDA', 'NFLX', 'PYPL', 'INTL', 'BABA', 'AMD', 'INTC', 'CRM','PYPL'}
if not Gainers_list.empty:
    stocks.update(Gainers_list['Symbol'].unique().tolist())
if not Loosers_list.empty:
    stocks.update(Loosers_list['Symbol'].unique().tolist())

col_stock, col_date, col_interval = st.columns(3)

with col_stock:
    selected_stock = st.selectbox('Select a stock for analysis', sorted(list(stocks)))

with col_date:
    START = st.date_input("Start date", datetime.date(2015, 1, 1))
    TODAY = date.today()

with col_interval:
    selected_interval = st.radio("Select Interval", options=['1d', '1wk'], horizontal=True)

# --- Data Fetching and Display ---
data_load_state = st.text(f"Loading data for {selected_stock}...")
data = load_data(selected_stock, selected_interval)
data = data.loc[START.strftime("%Y-%m-%d"):TODAY.strftime("%Y-%m-%d")].copy() # Filter by date
data_load_state.text(f"Loading data for {selected_stock}... done!")


tab1, tab2 = st.tabs(["Charts", "Raw Data"])

with tab2:
    st.subheader(f'Raw Data for {selected_stock}')
    st.write(data.tail(50))

with tab1:
    st.subheader("Historical Price Chart")
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series Stock Data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
    
    if not data.empty:
        plot_raw_data()
    else:
        st.warning("No data to plot.")

# --- Company Information ---
st.subheader(f"About {selected_stock}")
info = get_company_info(selected_stock)
if info is not None:
    latest_volume = data['Volume'].iloc[-1] if not data.empty else 'N/A'
    
    st.write(f"**Industry:** {info.get('Industry', 'N/A')}")
    st.write(f"**Sector:** {info.get('Sector', 'N/A')}")
    st.write(f"**Market Cap:** ${int(info.get('MarketCapitalization', 0)):,}")
    st.write(f"**Latest Volume:** {int(latest_volume):,}")
    st.write(f"**52 Week High:** {info.get('52WeekHigh', 'N/A')}")
    st.write(f"**52 Week Low:** {info.get('52WeekLow', 'N/A')}")
    
    with st.expander("Full Business Summary"):
        st.write(info.get('Description', 'No summary available.'))
else:
    st.warning("Could not retrieve company information.")


# --- Technical Analysis ---
with st.container():
    st.subheader('Momentum Analysis')
    metrics = {'EMAF', 'RSI', 'SMA', 'EMAM', 'EMAS'}
    selected_metric = st.multiselect('Choose indicators to analyze:', options=list(metrics), default=['SMA'])
    
    if not data.empty:
        data['RSI'] = ta.rsi(data.Close, length=15)
        data['EMAF'] = ta.ema(data.Close, length=20)
        data['EMAM'] = ta.ema(data.Close, length=100)
        data['EMAS'] = ta.ema(data.Close, length=150)
        data['SMA'] = ta.sma(data.Close, timeperiod=10)

        fig3 = go.Figure()
        fig3.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"))
        
        if 'SMA' in selected_metric:
            fig3.add_trace(go.Scatter(x=data.index, y=data['SMA'], name="SMA"))
        if 'EMAF' in selected_metric:
            fig3.add_trace(go.Scatter(x=data.index, y=data['EMAF'], name="EMAF"))
        if 'RSI' in selected_metric:
            fig3.add_trace(go.Scatter(x=data.index, y=data['RSI'], name="RSI"))
        if 'EMAM' in selected_metric:
            fig3.add_trace(go.Scatter(x=data.index, y=data['EMAM'], name="EMAM"))
        if 'EMAS' in selected_metric:
            fig3.add_trace(go.Scatter(x=data.index, y=data['EMAS'], name="EMAS"))

        fig3.update_layout(height=600, title_text='Stock Metrics')
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
        fig4.update_layout(height=300, title_text='Volume')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Cannot perform technical analysis, data is missing.")
        
# --- Investment Analysis (Comparison) ---
with st.container():
    st.subheader('Investment Comparison')
    analyzed_stocks = st.multiselect('Choose two stocks to compare:', options=sorted(list(stocks)), default=['AMZN', 'AAPL'])
    
    if len(analyzed_stocks) != 2:
        st.warning("Please select exactly two stocks for comparison.")
    else:
        stock1_data = load_data(analyzed_stocks[0], '1d')
        stock2_data = load_data(analyzed_stocks[1], '1d')

        if not stock1_data.empty and not stock2_data.empty:
            df1 = stock1_data['Adj Close'].to_frame().rename(columns={"Adj Close": analyzed_stocks[0]})
            df2 = stock2_data['Adj Close'].to_frame().rename(columns={"Adj Close": analyzed_stocks[1]})
            
            comparison_df = pd.concat([df1, df2], axis=1).pct_change().dropna()

            # Correlation Plot
            st.write('**Correlation of Daily Returns**')
            st.write('This plot shows how the daily returns of the two stocks move in relation to each other.')
            fig_corr = sns.jointplot(data=comparison_df, x=analyzed_stocks[0], y=analyzed_stocks[1], kind='scatter')
            st.pyplot(fig_corr)

            # Risk vs. Return Plot
            st.write('**Expected Return vs. Risk**')
            st.write('This plot shows the average daily return (Expected Return) against the standard deviation of daily returns (Risk).')
            rets = comparison_df.dropna()
            area = np.pi * 10
            
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(
                x=rets.mean(), 
                y=rets.std(), 
                mode='markers', 
                marker=dict(size=area * 2)
            ))
            fig_risk.update_layout(
                title='Expected Return vs. Risk',
                xaxis_title='Expected Return (Mean)',
                yaxis_title='Risk (Standard Deviation)',
                showlegend=False
            )
            for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
                fig_risk.add_annotation(x=x, y=y, text=str(label), showarrow=True, xanchor='left', yanchor='bottom')
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.error("Could not load data for one or both stocks to compare.")

# --- Prediction Section ---
# Note: The Keras/TensorFlow libraries might be heavy for a standard Streamlit app.
# Ensure your deployment environment has enough resources.
try:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    
    with st.container():
        st.header('Stock Predictions')
        if data.empty:
            st.warning("Cannot run prediction as no data was loaded.")
        else:
            st.write('This section uses a simple LSTM model to predict the next closing price based on historical data.')
            
            ticker_close = data[['Close']].copy()
            dataset = ticker_close.values
            training_data_len = int(np.ceil(len(dataset) * .95))

            with st.spinner("Preprocessing data for the model..."):
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)
                train_data = scaled_data[0:int(training_data_len), :]
                x_train, y_train = [], []
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i-60:i, 0])
                    y_train.append(train_data[i, 0])
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            with st.spinner("Training the LSTM model... This may take a moment."):
                model = Sequential()
                model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(64, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, batch_size=1, epochs=1)

            with st.spinner("Evaluating the model and making predictions..."):
                test_data = scaled_data[training_data_len - 60:, :]
                x_test, y_test = [], dataset[training_data_len:, :]
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)
                
                rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
                st.write(f'**Root Mean Squared Error (RMSE):** {rmse:.2f}')
                st.info('A lower RMSE means the predicted values are closer to the actual values.')

            # Prepare data for plotting
            train = data[:training_data_len]
            valid = data[training_data_len:].copy()
            valid['Predictions'] = predictions
            
            st.subheader("Prediction vs. Actual")
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Training Data'))
            fig_pred.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual Price'))
            fig_pred.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted Price'))
            fig_pred.update_layout(title_text='Model Prediction Results', xaxis_title='Date', yaxis_title='Close Price USD ($)')
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.subheader("Prediction Data")
            st.write(valid[['Close', 'Predictions']].tail(10))
            
except ImportError:
    st.warning("TensorFlow/Keras not installed. The prediction section is disabled. Please install with `pip install tensorflow`.")

