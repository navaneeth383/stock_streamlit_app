import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import date, timedelta
import matplotlib.pyplot as plt

# ----------------- UI Setup -----------------
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Prediction App")
st.markdown("Developed by **Gattu Navaneeth Rao**")

stock_name = st.text_input("Enter stock symbol (e.g., TCS.NS)", value="TCS.NS")
start_date = st.date_input("From", date(2020, 1, 1))
end_date = st.date_input("To", date.today())

if st.button("Train and Predict"):
    try:
        # 1. Download Data
        data = yf.download(stock_name, start=start_date, end=end_date)
        st.success("Data downloaded successfully.")
        
        # 2. Plot
        st.subheader("📊 Closing Price Chart")
        st.line_chart(data['Close'])

        # 3. Preprocess
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        # 4. Prepare for LSTM
        def create_dataset(data, time_step=60):
            X, Y = [], []
            for i in range(time_step, len(data)):
                X.append(data[i-time_step:i])
                Y.append(data[i])
            return np.array(X), np.array(Y)

        X, y = create_dataset(scaled_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # 5. Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        st.success("Model trained.")

        # 6. Predict next 7 days
        last_60_days = scaled_data[-60:]
        input_seq = last_60_days.reshape(1, 60, 1)
        preds = []

        for _ in range(7):
            pred = model.predict(input_seq)[0]
            preds.append(pred[0])
            input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)

        future_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

        st.subheader("📅 Predicted Prices for Next 7 Days")
        future_dates = [date.today() + timedelta(days=i) for i in range(1, 8)]
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_prices.flatten()})
        st.write(pred_df)

        st.subheader("📈 Price Forecast Chart")
        fig, ax = plt.subplots()
        ax.plot(pred_df['Date'], pred_df['Predicted Close'], marker='o', linestyle='--')
        st.pyplot(fig)

        # 7. Recommendation
        last_real = close_prices[-1][0]
        next_day_price = future_prices[0][0]
        if next_day_price > last_real:
            st.success(f"🔼 Predicted rise to ₹{next_day_price:.2f}. Consider HOLD or BUY.")
        else:
            st.warning(f"🔽 Predicted drop to ₹{next_day_price:.2f}. Consider SELL or WAIT.")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
