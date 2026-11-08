# Load the preprocessed data
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('all_data.csv', index_col=0)

print(df.head()) # Check loaded data

df.drop(['Month', 'Day', 'Hour', 'unique-key'], axis = 1, inplace = True)
temp_df = df

print(temp_df.head())

temp_df['gridID'] = (temp_df['gridID'].values / 48080)
temp_df['WindSpeed'] = (temp_df['WindSpeed'].values / 160)
temp_df['distance'] = (temp_df['distance'].values / 3561)
temp_df['direction'] = (temp_df['direction'].values / 360)
temp_df['Lat'] = temp_df['Lat'].values/81
temp_df['Long'] = (temp_df['Long'].values+137)/200

def load_data(stock, seq_len):
    data = stock.values
    sequence_length = seq_len + 1  # Because index starts at 0
    result = []

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = len(result) * 0.9  # Amount of data to train on
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))

    return [x_train, y_train, x_test, y_test]

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Input  # Add Input here
import math, time

x_train, y_train, x_test, y_test = load_data(temp_df[::-1], 14)
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)

model = Sequential()
# Define input layer
model.add(Input(shape=(14, 6)))  # Assuming 1 timestep and layers[0] features

for x in range(5):
    model.add(LSTM(units=14, return_sequences=True))  # No need for input_shape here
    model.add(Dropout(0.2))

# Last LSTM layer without return_sequences
model.add(LSTM(1, return_sequences=False))
model.add(Dropout(0.2))

# Dense layer using 'units' instead of 'output_dim'
model.add(Dense(units=1))
model.add(Activation("relu"))

# Compilation step
start = time.time()
model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
print("Compilation Time : ", time.time() - start)

model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.1, verbose=0)

trainScore = model.evaluate(x_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(x_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

model.save('hurricane.keras')
