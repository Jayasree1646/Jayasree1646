import Libraries

import numpy as np import pandas as pd import yfinance as yf
import matplotlib.pyplot as plt from sklearn.preprocessing import MinMaxScaler from sklearn.metrics
import mean_squared_error, mean_absolute_error, r2_score from statsmodels.tsa.arima.model
import ARIMA from fbprophet
import Prophet from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense, Dropout
Step 1: Data Collection

ticker = "AAPL"
start_date = "2010-01-01"
end_date = "2023-01-01"
data = yf.download(ticker, start=start_date, end=end_date) data.reset_index(inplace=True)

Step 2: Data Preprocessing Normalization

data['Date'] = pd.to_datetime(data['Date']) data.set_index('Date', inplace=True) data = data[['Close']] scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

Train-Test Split (80-20)

train_size = int(len(data_scaled) * 0.8) train_data = data_scaled[:train_size] test_data = data_scaled[train_size:]

Convert to Time Series Format (X: past N days, Y: next day)

def create_dataset(dataset, look_back=60):
X, Y = [], [] for i in range(look_back, len(dataset)): X.append(dataset[i- look_back:i, 0]) Y.append(dataset[i, 0])
return np.array(X), np.array(Y)
look_back = 60 X_train, y_train = create_dataset(train_data, look_back) X_test, y_test = create_dataset(test_data, look_back)
Reshape for LSTM (samples, timesteps, features)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Step 3: Model Building (LSTM)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1))) model.add(Dropout(0.2)) model.add(LSTM(units=50, return_sequences=False)) model.add(Dropout(0.2)) model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error') model.fit(X_train, y_train, epochs=20, batch_size=32)

Step 4: Predictions

train_predict = model.predict(X_train) test_predict = model.predict(X_test)

Inverse Scaling
train_predict = scaler.inverse_transform(train_predict) y_train = scaler.inverse_transform([y_train]) test_predict =
scaler.inverse_transform(test_predict) y_test = scaler.inverse_transform([y_test])

Step 5: Evaluation

rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
mae = mean_absolute_error(y_test[0], test_predict[:,0])) r2 = r2_score(y_test[0], test_predict[:,0]))
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R² Score: {r2}")
Step 6: Visualization

plt.figure(figsize=(12,6)) plt.plot(data.index[train_size+look_back:],
y_test[0], label='Actual Price') plt.plot(data.index[train_size+look_back:], test_predict[:,0], label='Predicted Price')
plt.title(f"{ticker} Stock Price Prediction (LSTM)") plt.xlabel('Date')
plt.ylabel('Price ($)') plt.legend() plt.show()

