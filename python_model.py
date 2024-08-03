import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler

today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=1825)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

df= yf.download('AAPL',start=start_date,end=end_date,progress=False)

print(df.head())

df=df.reset_index()

ma100 = df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()

#print(ma100)
#print(ma200)

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.show()

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_training.shape)
print(data_testing.shape)
print(data_testing)

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)
#print(data_training_array)

x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
	x_train.append(data_training_array[i-100:i])
	y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train),np.array(y_train)

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model=Sequential()

model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2)) 

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3)) 

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4)) 

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5)) 

model.add(Dense(units=1))

#print(model.summary())

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=1)
model.save('my_model2.h5')

print(data_testing.head())
print(data_training.tail(100))

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
print(final_df.head())

input_data=scaler.fit_transform(final_df)
print(input_data.shape)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
	x_test.append(input_data[i-100:i])
	y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
#print(x_test.shape)
#print(y_test.shape)

y_predicted=model.predict(x_test)
#print(y_predicted.shape)

#print(scaler.scale_)
scale_factor=1/(scaler.scale_)
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
