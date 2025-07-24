import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")

st.title("📈 Stock Price Predictor")
st.markdown("Predict next week's stock closing prices using Machine Learning")

# Sidebar
st.sidebar.header("🔧 Model Configuration")
ticker = st.text_input("Enter Stock Symbol (e.g. TCS.NS)", "TCS.NS")
start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.now())
look_back = st.sidebar.slider("Lookback Window (days)", 30, 120, 60)

# Get data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

try:
    df = load_data(ticker, start_date, end_date)
    st.write(f"Showing data from **{df.index.min().date()}** to **{df.index.max().date()}**")
    st.line_chart(df['Close'])

    # Preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    def create_dataset(data, look_back):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Predict next 7 days
    last_sequence = scaled_data[-look_back:]
    predictions = []
    for _ in range(7):
        input_seq = np.reshape(last_sequence, (1, look_back, 1))
        pred = model.predict(input_seq, verbose=0)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], [[pred[0][0]]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    next_dates = pd.date_range(end_date + timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({'Date': next_dates, 'Predicted Close': predictions.flatten()})
    st.subheader("📅 Next 7-Day Price Forecast")
    st.dataframe(forecast_df.set_index("Date"))

    # Trend recommendation
    if predictions[-1] > df['Close'].iloc[-1]:
        advice = "📈 **Hold** – Uptrend expected."
    else:
        advice = "📉 **Sell** – Downtrend likely."

    st.success(f"Recommendation: {advice}")

    # Plot forecast
    fig, ax = plt.subplots()
    ax.plot(df.index[-30:], df['Close'].tail(30), label="Historical")
    ax.plot(forecast_df['Date'], forecast_df['Predicted Close'], label="Forecast", color="orange")
    ax.set_title("Closing Price Forecast")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Something went wrong: {e}")

st.markdown("---")
st.markdown("👨‍💻 Developed by **Gattu Navaneeth Rao**")
