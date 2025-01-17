#Recurrent Neural Network

#Data preprocessing

#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

#importing training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#feature scaling

from sklearn.preprocessing import MinMaxScaler
sc  = MinMaxScaler(feature_range=(0,1))
scaled_training_set = sc.fit_transform(training_set)

#creating 60 timesteps adn 1 output data_structure

X_train = []
y_train = []

for i in range(120,1258):
    X_train.append(scaled_training_set[i-120:i,0])
    y_train.append(scaled_training_set[i,0])
X_train , y_train = np.array(X_train), np.array(y_train)

#reshaping

X_train = np.reshape(X_train, (X_train.shape[0] , X_train.shape[1] , 1))

#part-2 Building RNN

#importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#inititalisizing RNN
regressor = Sequential()

#adding LSTM and Dropout reguilarization

regressor.add(LSTM(units = 50, return_sequences = True , input_shape = (X_train.shape[1] , 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Output Layer
regressor.add(Dense(units = 1))

#compilation of rnn

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#training RNN with the training set

regressor.fit(X_train, y_train, epochs = 120, batch_size = 32)

#part 3

#inserting test dataset

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#concatinating dataset for prediction

total_data = pd.concat((dataset_train['Open'],dataset_test['Open']), axis=0)
inputs = total_data[len(total_data)-len(dataset_test)-120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(120, 140):
    X_test.append(inputs[i-120:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualizing the result
plt.plot(real_stock_price, color = 'red', label = "Real Stock Price")
plt.plot(predicted_stock_price, color = 'blue', label = "Predicted Stock Price")
plt.xlabel("Time")
plt.ylabel("Stock Market")
plt.show()



