import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime
import yfinance as yf
from datetime import date,timedelta
from sklearn.svm import SVR
import plotly.graph_objects as go


today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=1825)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

st.title('Stock Trend Prediction')

user_input=st.text_input("Enter Stock Ticker","TSLA")
df=yf.download(user_input,start=start_date,end=end_date,progress=False)
#st.write(df)
st.subheader('Statistical summary of %s'%(user_input))
st.subheader('Data from %s to %s'%(d2,d1))
st.write(df.describe())

# Closing Price Vs Time Chart
st.subheader("Closing Price Vs Time Chart")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Closing Price", line={'color':'#1f77b4'}))
fig.update_layout(
    title={
        'text': "",
        'x': 0.5,
        'y': 0.9,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="Price"
)
st.plotly_chart(fig)


# Closing Price Vs Time Chart with 100MA
st.subheader('Closing Price Vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=ma100, name="100MA"))
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Closing Price"))
fig.update_layout(
    title={
        'text': "",
        'x': 0.5,
        'y': 0.9,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="Price"
)
st.plotly_chart(fig)



# Closing Price Vs Time Chart with 100MA and 200MA
st.subheader('Closing Price Vs Time Chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=ma100, name="100MA",line=dict(color='white')))
fig.add_trace(go.Scatter(x=df.index, y=ma200, name="200MA" ,line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Closing Price", line=dict(color='#1f77b4')))
fig.update_layout(
    title={
        'text': "",
        'x': 0.5,
        'y': 0.9,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis_title="Year",
    yaxis_title="Price"
)
st.plotly_chart(fig)

  
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)

#Load my model
model=load_model('keras_model.h5')

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
	x_test.append(input_data[i-100:i])
	y_test.append(input_data[i,0])


x_test, y_test=np.array(x_test), np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test=y_test*scale_factor



# Prediction Vs Original
st.subheader('Prediction Vs Original')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-len(y_test):], y=y_test, name='Original Price'))
fig.add_trace(go.Scatter(x=df.index[-len(y_test):], y=y_predicted.reshape(-1), name='Predicted Price'))
fig.update_layout(
    title="",
    xaxis_title="Year",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True
)
st.plotly_chart(fig)


#Prediction Graph
df['Year'] = pd.DatetimeIndex(df.index).year
# Create an array of the independent variables (features)
X = np.array(df['Year']).reshape(-1, 1)

# Create an array of the dependent variable (target)
y = np.array(df['Close'])

# Create a Support Vector Regression model with a radial basis function kernel
model = SVR(kernel='rbf', C=1e3, gamma=0.1)

# Train the model on the data
model.fit(X, y)

# Make predictions for the previous 3 years and the next 3 years


previous_years = np.array(range(df['Year'].min() - 3, df['Year'].min())).reshape(-1, 1)
future_years = np.array(range(df['Year'].max() + 1, df['Year'].max() + 4)).reshape(-1, 1)
all_years = np.concatenate((previous_years, X, future_years), axis=0)

predictions = model.predict(all_years)



st.subheader('Future Stock Price Prediction Graph')
fig = go.Figure()
fig.add_trace(go.Scatter(x=all_years.flatten(), y=predictions, name='Predictions'))
fig.update_layout(
    title='',
    xaxis_title='Year',
    yaxis_title='Stock Price'
)
st.plotly_chart(fig)