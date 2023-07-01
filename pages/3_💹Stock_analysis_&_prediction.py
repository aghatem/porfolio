# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import yfinance as yf
import requests
import pandas as pd
import pandas_ta as ta
from bs4 import BeautifulSoup
import datetime
import seaborn as sns

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# yfinance documentation https://pypi.org/project/yfinance/


gianers_url =  'https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/'
losers_url =  'https://www.tradingview.com/markets/stocks-usa/market-movers-losers/'




def get_top_movers_list(url):
    response = requests.get(url)
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
        #company_name = cells[0].find('sup').previous_sibling.strip()
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

    df = pd.DataFrame(movers)
    return df

Gainers_list = get_top_movers_list(gianers_url)
Loosers_list = get_top_movers_list(losers_url)


st.title('Stock Forecast App')

st.write('This web app will provide you the information needed to analyze the stock market and will calculate the key technical indicators neneded to analyse the stock then will predict the next day closing value.')
st.write('it will calculate the Simple Moving Average (SMA), Exponential Moving Average (EMA), Bollinger Bands (BB), Stochastic Oscillator (STOCH), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD) & Volume Weighted Average Price (VWAP)')
st.write('It will rely on the historical stock value from yfinance API (in addition to other information like yearly dividend percentage, Price to Earning Ratio and details on the company). ')
st.write('This web app will help you to analyze the daily top 100 tickers gainers and loosers as it performs HTML parsing from www.tradingview.com to provide you with the daily lists')
with st.expander("Click to see todays top movers "):

	list =('Gainers','Loosers')
	selected_movers =  st.selectbox('To get the list, please select the Top gainers or loosers for today :' , list)
	if selected_movers == 'Gainers':
			st.write(Gainers_list)
	else:
			st.write(Loosers_list)

# load tickers
stocks = {'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'NFLX', 'PYPL', 'INTL', 'BABA', 'AMD', 'INTC', 'CRM','PYPL', 'ATVI', 'TTD', 'EA','MTCH', 'ZG'}
gainers_stocks = Gainers_list['Symbol'].unique().tolist()
loosers_stocks = Loosers_list['Symbol'].unique().tolist()  
stocks.update(gainers_stocks)
stocks.update(loosers_stocks)
# user inputs 
selected_stock = st.selectbox('Select dataset for analysis & prediction', stocks)



    

START = st.date_input("Start date",datetime.date(2023, 1,1 ))
def update_date():
    new_date = st.date_input('Select a new date')
    st.session_state['date'] = new_date

button = st.button('Update Date', on_click=update_date)

if 'date' not in st.session_state:
    st.session_state['date'] = date(2018, 1,1 )

if button:
    st.write(f'Date updated.')
TODAY = date.today().strftime("%Y-%m-%d")



#@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY, interval=selected_interval)
    
    return data

interval = ['1d','1wk']
selected_interval = st.radio("Select an option", options = interval, horizontal=True)
tab1 , tab2 = st.tabs([ "Charts","Data"])

with tab2:
	
	data = load_data(selected_stock)
	
	cm = sns.light_palette("green", as_cmap=True)
	data.style.background_gradient(cmap=cm)
	st.write(data.tail(50))


with tab1:

	st.subheader("Historical data ")
	
	# Plot raw data
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data.index, y=data['Open'], name="stock_open"))
		fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="stock_close"))
		fig.layout.update(title_text='Time Series stock data', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
		
	plot_raw_data()


ticker = yf.Ticker(str(selected_stock))

st.subheader(selected_stock)
st.write(selected_stock,' company is based in ', ticker.info['country'], ' , its main industry is ',ticker.info['industry'],' and its website is ',ticker.info['website'])
st.write('The trading volume is',ticker.info['marketCap'] ,' and the market cap is ',ticker.info['volume'])
st.write('The 52 weeks trading high is ',ticker.info['fiftyTwoWeekHigh'], ' and the 52 weeks trading low is ',ticker.info['fiftyTwoWeekLow'])
st.write(ticker.info['longBusinessSummary'])
with st.expander("For a closer look to the stock data, click here !"):
	temp = pd.DataFrame.from_dict(ticker.info, orient="index")
	temp.reset_index(inplace=True)
	temp.columns = ["Attribute", "Recent"]
	st.write(temp) 
	




	
with st.container():
	st.subheader('Stock indicators')

	metrics = {'EMAF','RSI','SMA','EMAM','EMAS'}
	selected_metric = st.multiselect('Please chose the indicators you want to analyze: ', options = metrics, default= ['SMA'] )

	data['RSI']=ta.rsi(data.Close, length=15)
	data['EMAF']=ta.ema(data.Close, length=20)
	data['EMAM']=ta.ema(data.Close, length=100)
	data['EMAS']=ta.ema(data.Close, length=150)
	data['SMA'] = ta.sma(data.Close,timeperiod=10)

	
	fig3 = go.Figure()
	fig3.add_trace(go.Candlestick(x=data.index,open=data['Open'],high=data['High'],low=data['Low'],
	close=data['Close'], name="Price"))
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

	fig3.update_layout(height=600, width=800, title_text='Stock Metrics')
	st.plotly_chart(fig3)

	fig4=go.Figure()
	fig4.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
	fig4.update_layout(height=300, width=800, title_text='Volume')
	st.plotly_chart(fig4)

with st.container():
	st.subheader('Investment Analysis')

	analyzed_stocks = st.multiselect('Please chose the stocks you want to analyze: ', options = stocks )
	if analyzed_stocks == []:
		st.write("Please select max 2 stocks.")
	
	elif len(analyzed_stocks) == 1:
		st.write("Please chose another stock to analyze.")
	elif len(analyzed_stocks) >2:
		st.write("You slected more than 2, please select max 2 stocks.")
	else:
		st.write(analyzed_stocks[0])
		df1=load_data(analyzed_stocks[0])['Adj Close'].to_frame()
		df1['Adj Close'].fillna(0)
		df1 = df1.rename(columns={"Adj Close": analyzed_stocks[0]+'_Adj Close'})
		df2=load_data(analyzed_stocks[1])['Adj Close'].to_frame()
		df2['Adj Close'].fillna(0)
		df2 = df2.rename(columns={"Adj Close": analyzed_stocks[1]+'_Adj Close'})
		df_3 = pd.concat([df1, df2], axis=1).pct_change()
		st.write(df_3)
		# plot the correlation between 2 stocks
		st.write('Correlation is a statistic that measures the degree to which two variables move in relation to each other. ')
		st.write('In below jointplot chart, we compare the daily return of the two stocks you have selected to check how they correlate.')
		jointplot=sns.jointplot(df_3, x=analyzed_stocks[0]+'_Adj Close',y=analyzed_stocks[1]+'_Adj Close', kind='scatter')
		#st.plotly_chart(jointplot)
		# Get the jointplot data for plotting
		x = jointplot.ax_joint.collections[0].get_offsets()[:, 0]
		y = jointplot.ax_joint.collections[0].get_offsets()[:, 1]

		scatter_plot = go.Scatter(x=x, y=y, mode='markers')
		# Create a Plotly layout
		layout = go.Layout(title="Correlation using Jointplot",xaxis=dict(title=analyzed_stocks[0]),yaxis=dict(title=analyzed_stocks[1]))
		fig6 = go.Figure(data=[scatter_plot], layout=layout)
		st.plotly_chart(fig6)
	
		# Calculate risk
		
		rets = df_3.dropna()
		st.write('Risk is the uncertainty of the return on an investment. It is measured by the standard deviation of the returns. A higher standard deviation indicates that the returns are more volatile, and therefore riskier.')
		st.write('Return is the income that an investment generates. It is measured by the mean of the returns. A higher mean indicates that the returns are higher on average.')
		st.write('In below chart the risk and return of the stocks you selected are calculated using below formulas:')
		st.write(' - Risk: Standard deviation of returns')
		st.write(' - Return: Mean of returns')
		area = np.pi * 10
		# Create a Plotly scatter plot & layout
		scatter_plot = go.Scatter(x=rets.mean(),y=rets.std(),mode='markers',marker=dict(size=area),	)
		layout = go.Layout(title='Expected Return vs. Risk',xaxis=dict(title='Expected return'),yaxis=dict(title='Risk'),)
		fig7 = go.Figure(data=[scatter_plot], layout=layout)

		# Annotate the points with column labels
		for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
			fig7.add_annotation(x=x,y=y,text=str(label),xanchor='left',showarrow=True,arrowhead=1,arrowsize=1,arrowcolor='lightgreen',ax=50,ay=-50,font=dict(family="Arial",size=18,color="black")	)
		# Display the Plotly figure in Streamlit
		st.plotly_chart(fig7)



#---------------------------------------------
# #data = yf.download("SPY AAPL", start="2017-01-01", end="2017-04-30")
# yf.download(tickers = "SPY AAPL",  # list of tickers
#             period = "1y",         # time period
#             interval = "1d",       # trading interval
#             prepost = False,       # download pre/post market hours data?
#             repair = True)         # repair obvious price errors e.g. 100x?

# pandas_datareader override

# from pandas_datareader import data as pdr

# import yfinance as yf
# yf.pdr_override() # <== that's all it takes :-)

# # download dataframe
# data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
# #Market cap -> aapl.info["marketCap"], current volume -> aapl.info["volume"]
# data = yf.download("AMZN AAPL GOOG", start="2017-01-01",
# #                     end="2017-04-30", group_by='tickers')
# aapl_historical = aapl.history(period="max", interval="1wk")

# aapl.info["fiftyTwoWeekHigh"]
# dayHigh,dayLow,fiftyTwoWeekHigh,fiftyTwoWeekLow
# # Price to Earning Ratio (P/E) 
# aapl = yf.Ticker("aapl")
# aapl.info['forwardPE']

# get the yearly dividend % -> aapl.info['dividendRate']
#-------------------------------------------------------------------------
## To get company info 
# for ticker in tickers_list:
#     ticker_object = yf.Ticker(ticker)

#     #convert info() output from dictionary to dataframe
#     temp = pd.DataFrame.from_dict(ticker_object.info, orient="index")
#     temp.reset_index(inplace=True)
#     temp.columns = ["Attribute", "Recent"]
    
#     # add (ticker, dataframe) to main dictionary
#     tickers_data[ticker] = temp

# combined_data = pd.concat(tickers_data)
# combined_data = combined_data.reset_index()
# del combined_data["level_1"] # clean up unnecessary column
# combined_data.columns = ["Ticker", "Attribute", "Recent"] # update column names
# combined_data

#-------------------------------------------------------------------------

## compare 2 companies on a specific atribute
# employees = combined_data[combined_data["Attribute"]=="fullTimeEmployees"].reset_index()
# del employees["index"] # clean up unnecessary column



#----------------------------------------------------------------

# Predict forecast with Prophet.
# df_train = data[['Date','Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)

# # Show and plot forecast
# st.subheader('Forecast data')
# st.write(forecast.tail())
    
# st.write(f'Forecast plot for {n_years} years')
# fig1 = plot_plotly(m, forecast)
# st.plotly_chart(fig1)

# st.write("Forecast components")
# fig2 = m.plot_components(forecast)
## st.write(fig2)
