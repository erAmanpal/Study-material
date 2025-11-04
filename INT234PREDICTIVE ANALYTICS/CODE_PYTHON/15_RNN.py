import pandas as pd
# Step 1: Load and preprocess data
df = pd.read_csv('stock_prices.csv')
#Reshapes the data into a column vector for scaling and modeling.
prices = df['close'].values.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
# Step 2: Normalize prices
#Normalizing helps the model learn patterns more effectively and converge faster.
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

import numpy as np
# Step 3: Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 30  # Use last 30 minutes to predict next
X, y = create_sequences(prices_scaled, seq_len)
X = X.reshape(-1, seq_len, 1) #(samples, time steps, features)
y = y.reshape(-1, 1)

# Step 4: Train/test split
#Splits data into training (80%) and testing (20%) sets.
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense
# Step 5: Build RNN model
model = Sequential([
    Input(shape=(seq_len, 1)),          #30 time steps, 1 feature.
    SimpleRNN(32, activation='tanh'),   #Core RNN layer with 32 units.
    Dense(1)        #single value — the predicted next minute’s price.
])
##adam optimizer: Efficient gradient descent.
##mse loss: Measures how far predictions are from actual prices.
##epochs=10: Trains the model 10 times over the dataset.
##batch_size=32: Processes 32 samples at a time.
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)


# Step 6: Predict next minute price using the trained RNN
##prices_scaled[-seq_len:] selects the most recent 30 minutes (if seq_len = 30).
##.reshape(1, seq_len, 1) reshapes it into the format expected by the RNN
## (samples, time steps, features)
last_sequence = prices_scaled[-seq_len:].reshape(1, seq_len, 1)
next_scaled = model.predict(last_sequence)
# Now you have the actual predicted price in real-world units
next_price = scaler.inverse_transform(next_scaled)
print("Predicted next minute close price:", next_price[0][0])

import matplotlib.pyplot as plt
# Step 7: Plot actual vs predicted (optional)
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.title("RNN: Minute-Level Stock Price Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.show()
