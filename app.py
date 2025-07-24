import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Prediction with LSTM")
st.markdown("Developed by **Gattu Navaneeth Rao**")

# --- Sidebar ---
st.sidebar.header("Model Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., TCS.NS)", value="TCS.NS")
start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=365 * 2))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
epochs = st.sidebar.slider("Epochs", 10, 100, 50)
future_days = 7

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# --- Fetch data ---
try:
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        st.warning("No data found. Please check the stock symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.write(f"### Closing Price for {ticker}")
st.line_chart(df['Close'])

# --- Preprocess ---
data = df.filter(['Close'])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# --- LSTM Model ---
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0)

# --- Predict next 7 days ---
last_60_days = scaled_data[-60:]
X_test = []
X_test.append(last_60_days[:, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = []

for _ in range(future_days):
    pred = model.predict(X_test)
    predicted_price.append(pred[0][0])
    X_test = np.append(X_test[:, 1:, :], [[[pred[0][0]]]], axis=1)

predicted_price = scaler.inverse_transform(np.array(predicted_price).reshape(-1, 1))
dates = [df.index[-1] + timedelta(days=i + 1) for i in range(future_days)]

# --- Show results ---
st.subheader("📅 Predicted Closing Prices (Next 7 Days)")
forecast_df = pd.DataFrame({'Date': dates, 'Predicted Close': predicted_price.flatten()})
st.write(forecast_df.set_index('Date'))
st.line_chart(forecast_df.set_index('Date'))

# --- Recommendation ---
last_real_price = df['Close'].iloc[-1]
next_day_price = predicted_price[0][0]

st.subheader("📌 Model Suggestion")
if next_day_price > last_real_price:
    st.success(f"Predicted price ↑ from ₹{last_real_price:.2f} to ₹{next_day_price:.2f}. Suggestion: **BUY / HOLD**")
else:
    st.warning(f"Predicted price ↓ from ₹{last_real_price:.2f} to ₹{next_day_price:.2f}. Suggestion: **WAIT / SELL**")
