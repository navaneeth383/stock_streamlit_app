import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ----------------- Setup -----------------
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

# ----------------- Header -----------------
st.title("📊 Stock Price Prediction App")
st.markdown("Developed by **Gattu Navaneeth Rao**")

# ----------------- User Inputs -----------------
stock_symbol = st.text_input("Enter Stock Symbol (e.g. TCS.NS)", "TCS.NS")
start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=180))
end_date = st.date_input("End Date", value=datetime.now())

model_type = st.selectbox("Select ML Model", ["Random Forest", "Linear Regression"])

if st.button("Predict"):
    try:
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for the given symbol. Please try another.")
            st.stop()

        st.success(f"Fetched {len(df)} rows of data.")
        st.dataframe(df.tail())

        # Preprocessing
        data = df[['Close']]
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        # Create sequences
        X, y = [], []
        for i in range(10, len(data_scaled)):
            X.append(data_scaled[i-10:i])
            y.append(data_scaled[i])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], -1))  # Flatten

        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

        model.fit(X, y)

        # Predict next 7 days
        last_data = data_scaled[-10:].reshape(1, -1)
        predictions = []
        for _ in range(7):
            pred = model.predict(last_data)[0]
            predictions.append(pred)
            last_data = np.roll(last_data, -1)
            last_data[0, -1] = pred

        future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(7)]
        pred_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": pred_prices})
        st.subheader("📆 Next 7 Days Forecast")
        st.dataframe(forecast_df)

        # Advice
        last_real = df['Close'].iloc[-1]
        if pred_prices[0] > last_real:
            st.success(f"Predicted price rise from ₹{last_real:.2f} to ₹{pred_prices[0]:.2f}. Consider HOLD/BUY.")
        else:
            st.warning(f"Predicted price may drop from ₹{last_real:.2f} to ₹{pred_prices[0]:.2f}. Consider SELL or WAIT.")

        # Plotting
        st.subheader("📉 Forecast Chart")
        plt.figure(figsize=(10, 4))
        plt.plot(df.index, df['Close'], label='Historical')
        plt.plot(forecast_df["Date"], forecast_df["Predicted Close"], label='Forecast', marker='o')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
