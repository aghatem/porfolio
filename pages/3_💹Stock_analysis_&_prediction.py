
import streamlit as st
from datetime import date
import yfinance as yf
import requests
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import seaborn as sns
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from plotly import graph_objs as go
#from keras.models import Sequential
#from keras.layers import LSTM
#import tensorflow as tf
#from keras.models import Model
#from keras.layers import Dense, LSTM, Input, concatenate
from sklearn.preprocessing import MinMaxScaler



# yfinance documentation https://pypi.org/project/yfinance/


gianers_url =  'https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/'
losers_url =  'https://www.tradingview.com/markets/stocks-usa/market-movers-losers/'
mofta7='USNMZO47I5X7NDKV'



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

st.write('This web app will provide you the information needed to analyze the stock market and will rely on prohet machine learning model to predict the estimated value for a after a certain period.')
st.write('It will rely on the historical stock value from yfinance API (in addition to other information like yearly dividend percentage, Price to Earning Ratio and details on the company). ')
st.write('This web app will help you to analyze the daily top 100 tickers gainers and loosers as it performs HTML parsing from tradingview.com to provide you with the daily lists')
st.write('The user will be able to compare two stocks and estimate the potential risk and revenue from each stock, to guide the user to invest in a stock will be able to rely on ChatGPT to get more information about the stock.')
st.write('Eventually the historical prices will be fed into a Machine Learning Tensorflow model to be trained and then predict the closing value of the selected stock.')
with st.expander("Click to see todays top movers "):

	list =('Gainers','Loosers')
	selected_movers =  st.selectbox('To get the list, please select the Top gainers or loosers for today :' , list)
	if selected_movers == 'Gainers':
			st.write(Gainers_list)
	else:
			st.write(Loosers_list)

# load tickers
stocks = {'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'META', 'TSLA', 'NVDA', 'NFLX', 'PYPL', 'INTL', 'BABA', 'AMD', 'INTC', 'CRM','PYPL', 'ATVI', 'TTD', 'EA','MTCH', 'ZG'}
gainers_stocks = Gainers_list['Symbol'].unique().tolist() 
loosers_stocks = Loosers_list['Symbol'].unique().tolist()  
stocks.update(gainers_stocks)
stocks.update(loosers_stocks)
# user inputs 

col_stock, col_date, col_button = st.columns(3)

with col_stock:
	selected_stock = st.selectbox('Select a stock for analysis', stocks)


 

with col_date:
	START = st.date_input("Start date",datetime.date(2013, 1,1 ))
	def update_date():
		new_date = st.date_input('Select a new date')
		st.session_state['date'] = new_date
		button = st.button('Update Date', on_click=update_date)
		if button:
			st.write(f'Date updated.')
		
		if 'date' not in st.session_state:
			st.session_state['date'] = date(2015, 1,1 )
TODAY = date.today().strftime("%Y-%m-%d")
with col_button:
	
	interval = ['1d','1wk']

	selected_interval = st.radio("Select an option", options = interval, horizontal=True)




#@st.cache_data
def load_data(ticker):
   
    # read data from yfinance
    data = yf.download(ticker, START, TODAY, interval=selected_interval)
    return data


tab1 , tab2 = st.tabs([ "Charts","Data"])

with tab2:
	
	
	data = load_data(selected_stock)
	
	cm = sns.light_palette("green", as_cmap=True)
	#data.style.background_gradient(cmap=cm)
	st.write(data.tail(50))
	st.write(len(data))
	st.write(data.columns.tolist())


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

col1, col2 = st.columns(2)
with col1:
	ticker = yf.Ticker(str(selected_stock))
	st.write(ticker.info['country'])
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
with col2:
	
	st.subheader('More from ChatGPT plugin: ')
	from langchain import OpenAI
	import os
	gpt_mofta7 = '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'
	os.environ["OPENAI_API_KEY"] =gpt_mofta7

	from langchain.llms import OpenAI
	llm = OpenAI(temperature=0.9)  # model_name="text-davinci-003"
	text = st.text_area('Hi there! Im your assistant, and will provide you answers powered by ChatGPT ',value='Hi ChatGPT, How is stock '+ selected_stock +" is performing this week?")
	st.write('## ChatGP: ')
	st.markdown(llm(text))



	
with st.container():
	st.subheader('Momentum	 analysis')
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
	default_selection = ['AMZN', 'AAPL']
	analyzed_stocks = st.multiselect('Please chose the stocks you want to analyze: ', options = stocks,default=['AMZN', 'AAPL'] )
	if analyzed_stocks == []:
		st.write("Please select up to 2 stocks.")
	
	elif len(analyzed_stocks) == 1:
		st.write("Please chose another stock to analyze.")
	elif len(analyzed_stocks) >2:
		st.write("You selected more than 2, please select maximum 2 stocks.")
	else:
		df1=load_data(analyzed_stocks[0])['Adj Close'].to_frame()
		df1['Adj Close'].fillna(0)
		df1 = df1.rename(columns={"Adj Close": analyzed_stocks[0]+'_Adj Close'})
		df2=load_data(analyzed_stocks[1])['Adj Close'].to_frame()
		df2['Adj Close'].fillna(0)
		df2 = df2.rename(columns={"Adj Close": analyzed_stocks[1]+'_Adj Close'})
		df_3 = pd.concat([df1, df2], axis=1).pct_change()


		# plot the correlation between 2 stocks
		st.write('Correlation is a statistic that measures the degree to which two variables move in relation to each other. ')
		st.write('In below jointplot chart, we compare the daily return of the two stocks you have selected to check how they correlate.')
		jointplot=sns.jointplot(df_3, x=analyzed_stocks[0]+'_Adj Close',y=analyzed_stocks[1]+'_Adj Close', kind='scatter')
		
		# Get the jointplot data for plotting
		x = jointplot.ax_joint.collections[0].get_offsets()[:, 0]
		y = jointplot.ax_joint.collections[0].get_offsets()[:, 1]

		scatter_plot = go.Scatter(x=x, y=y, mode='markers')
		# Create a Plotly layout
		layout = go.Layout(title="Correlation using Jointplot",xaxis=dict(title=analyzed_stocks[0]),yaxis=dict(title=analyzed_stocks[1]))
		fig6 = go.Figure(data=[scatter_plot], layout=layout)
		#st.plotly_chart(fig6)
		st.set_option('deprecation.showPyplotGlobalUse', False)
		st.pyplot()
	
		# Calculate risk
		rets = df_3.dropna()

		area = np.pi * 10
		# Create a Plotly scatter plot & layout
		scatter_plot = go.Scatter(x=rets.mean(),y=rets.std(),mode='markers',marker=dict(size=area),	)
			
		layout = go.Layout(title='Expected Return vs. Risk',xaxis=dict(title='Expected return'),yaxis=dict(title='Risk'),)
		fig7 = go.Figure(data=[scatter_plot], layout=layout)

		# Annotate the points with column labels
		for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
			fig7.add_annotation(x=x, y=y, text=str(label),xanchor='left',showarrow=True,arrowhead=1,arrowsize=1,arrowcolor='lightgreen',ax=50,ay=-50,font=dict(family="Arial",size=18,color="black")	)
		# Display the Plotly figure in Streamlit
		st.plotly_chart(fig7)


with st.container():
	st.header('Stock Predictions')
	st.write('In the following section we will use one of the Tensorflow ML models to predict the closing price of the stock you have used. The LSTM model will rely on RNN (Recursive Neural Network) in the prediction and the RNN will be formed of 4 layers. We will then split the historical stock price data into 2 sets. We chose to use 95% of the data to train the model and 5% to evaluate the model and measure the accuracy of the predicted values.')
	
	st.write('You have selected stock', selected_stock, ' which will be considered in the prediction')
	ticker_close = data['Close'].copy()
	
	with st.spinner("Preprocessing the dataset and preparing your data in a suitable format for training a deep learning RNN model..."):
		# Convert the dataframe to a numpy array
		dataset = ticker_close.values
		# Get the number of rows to train the model on
		training_data_len = int(np.ceil( len(dataset) * .95 ))
		# Scale the data
		from sklearn.preprocessing import MinMaxScaler
		reshaped_dataset = dataset.reshape(-1, 1)
		scaler = MinMaxScaler(feature_range=(0,1))
		scaled_data = scaler.fit_transform(reshaped_dataset)

		
		# Create the training data set 
		# Create the scaled training data set
		train_data = scaled_data[0:int(training_data_len), :]
		# Split the data into x_train and y_train data sets
		x_train = []
		y_train = []

		for i in range(60, len(train_data)):
			x_train.append(train_data[i-60:i, 0])
			y_train.append(train_data[i, 0])
			if i<= 61:
				print(x_train)
				print(y_train)
				print()
				
		# Convert the x_train and y_train to numpy arrays 
		x_train, y_train = np.array(x_train), np.array(y_train)

		# Reshape the data
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
		
	with st.spinner("Training the LSTM model with 95% of the dataset..."):
		# x_train.shape
		from keras.models import Sequential
		from keras.layers import Dense, LSTM

		# Build the LSTM model
		model = Sequential()
		model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
		model.add(LSTM(64, return_sequences=False))
		model.add(Dense(25))
		model.add(Dense(1))

		# Compile the model
		model.compile(optimizer='adam', loss='mean_squared_error')

		# Train the model
		model.fit(x_train, y_train, batch_size=1, epochs=1)
		#import keras.utils.plot_model
		
		#fig99 = keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
		
	with st.spinner("Evaluating the model & calculating the root mean squared error (RMSE)"):
		# Create the testing data set
		# Create a new array containing scaled values 
		test_data = scaled_data[training_data_len - 60: , :]
		# Create the data sets x_test and y_test
		x_test = []
		y_test = reshaped_dataset[training_data_len:, :]
		for i in range(60, len(test_data)):
			x_test.append(test_data[i-60:i, 0])
			
		# Convert the data to a numpy array
		x_test = np.array(x_test)

		# Reshape the data
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

		# Get the models predicted price values 
		predictions = model.predict(x_test)
		predictions = scaler.inverse_transform(predictions)

		# Get the root mean squared error (RMSE)
		rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
		st.write('The rmse is', rmse, ' a lower RMSE means that the predicted values are closer to the actual values.')

	with st.spinner('Ploting the training and the model prediction '):
		
		
		plot_tab , data_tab = st.tabs([ "Prediction plot","Prediction Data"])
		
		with data_tab:
			# Plot the data
			train = data[:training_data_len]
			valid = data[training_data_len:]
			valid['Predictions'] = predictions
			results = valid[['Close','Predictions']].copy()
			st.write(results.tail(20))

		with plot_tab:
			# Visualize the data
			fig8, ax = plt.subplots(figsize=(16,6))
			#ax.set_title(selected_stock, 'model')
			ax.set_xlabel('Date', fontsize=18)
			ax.set_ylabel('Close Price USD ($)', fontsize=18)
			ax.plot(train['Close'])
			ax.plot(valid[['Close', 'Predictions']])
			ax.legend(['Train', 'Eval', 'Predictions'], loc='lower right')
			st.pyplot(fig8)
	st.write('The predicted closing value is: ', round(results.iloc[len(results)-1,1],2), ' $')
	st.write('There are numerous ways to improve the model accuracy. For instance the length of the dataset, it is recommended to change the start date you selected at the beginning to increase the model performance and accuracy features selection. There are other techniques to improve the model accuracy which are not in the scope of this exercise, like increasing the number of layers in the RNN model architecture, optimizing the number of epochs, using multiple models, apply regularization to prevent overfitting, utilize cross-validation techniques to obtain a more reliable estimate of the models performance.')











