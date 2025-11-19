# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas_ta as ta
import os

# -------------------------------
# Helper Functions
# -------------------------------

def load_data(symbol, start="2010-01-01", end="2024-12-31"):
    df = yf.download(symbol, start=start, end=end)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['Signal'] = macd['MACDs_12_26_9']

    df.dropna(inplace=True)
    return df


def scale_data(df, features):
    scaler = MinMaxScaler()
    data = df[features].values
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def generate_signals(y_true, y_pred):
    signals = [0]  # First day no signal
    for i in range(1, len(y_pred)):
        signals.append(1 if y_pred[i] > y_true[i-1] else -1)
    return signals


def backtest(prices, signals, initial_capital=100000):
    capital = initial_capital
    shares = 0
    portfolio_values = []

    for i in range(len(prices)):
        price = prices[i]
        signal = signals[i]

        if signal == 1 and capital > 0:
            shares = capital / price
            capital = 0
        elif signal == -1 and shares > 0:
            capital = shares * price
            shares = 0

        portfolio_values.append(capital + shares * price)

    return portfolio_values


def forecast_future(model, scaled_data, scaler, days=30, seq_length=60):
    last_sequence = scaled_data[-seq_length:]
    future_predictions_scaled = []

    for _ in range(days):
        X_input = last_sequence.reshape(1, seq_length, scaled_data.shape[1])
        pred_scaled = model.predict(X_input, verbose=0)[0][0]

        row = np.zeros((scaled_data.shape[1],))
        row[0] = pred_scaled
        future_predictions_scaled.append(row.copy())

        last_sequence = np.vstack([last_sequence[1:], row])

    future_real = scaler.inverse_transform(np.array(future_predictions_scaled))[:, 0]
    return future_real


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📈 Stock Price Prediction & Analysis with LSTM")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
df = load_data(symbol)
st.subheader(f"Historical Data for {symbol}")
st.dataframe(df.tail())

# Features and scaling
features = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal"]
scaled_data, scaler = scale_data(df, features)
X, y = create_sequences(scaled_data)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# -------------------------------
# Load Saved Keras Model
# -------------------------------

model_path = "models/lstm_saved_model.keras"

if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("✅ Keras model loaded successfully!")
else:
    st.error(f"❌ Model not found at '{model_path}'")
    st.write("Current working directory:", os.getcwd())
    st.write("Files here:", os.listdir("."))
    st.write("Files in models/:", os.listdir("models"))
    st.stop()


# -------------------------------
# Predictions
# -------------------------------

y_pred = model.predict(X_test, verbose=0)

y_test_real = scaler.inverse_transform(
    np.hstack([y_test.reshape(-1,1), np.zeros((y_test.shape[0], len(features)-1))])
)[:,0]

y_pred_real = scaler.inverse_transform(
    np.hstack([y_pred, np.zeros((y_pred.shape[0], len(features)-1))])
)[:,0]


signals = generate_signals(y_test_real, y_pred_real)
portfolio_curve = backtest(y_test_real, signals)
final_value = portfolio_curve[-1]
roi = (final_value - 100000) / 100000 * 100


# -------------------------------
# Visualizations
# -------------------------------

st.subheader("Backtesting Results")
st.write(f"Initial Capital: ₹100,000")
st.write(f"Final Portfolio Value: ₹{final_value:.2f}")
st.write(f"ROI: {roi:.2f}%")


# Actual vs Predicted
st.subheader("📊 Actual vs Predicted Prices & Buy/Sell Signals")
plt.figure(figsize=(12,5))
plt.plot(y_test_real, label="Actual")
plt.plot(y_pred_real, label="Predicted")
plt.scatter(np.where(np.array(signals)==1)[0], y_test_real[np.where(np.array(signals)==1)[0]], marker="^", color="green", label="BUY")
plt.scatter(np.where(np.array(signals)==-1)[0], y_test_real[np.where(np.array(signals)==-1)[0]], marker="v", color="red", label="SELL")
plt.legend()
st.pyplot(plt.gcf())


# Portfolio curve
st.subheader("💹 Portfolio Equity Curve")
plt.figure(figsize=(12,5))
plt.plot(portfolio_curve, label="Portfolio Value")
plt.legend()
st.pyplot(plt.gcf())


# Future forecast
st.subheader("🔮 30-Day Future Forecast")
future_30 = forecast_future(model, scaled_data, scaler)
plt.figure(figsize=(12,5))
plt.plot(future_30, marker="o", label="Future Prices")
plt.legend()
st.pyplot(plt.gcf())

st.success("✅ Dashboard Ready!")
