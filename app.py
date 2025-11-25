############################################################
# FIX MAC M1/M2/M3 TENSORFLOW DEADLOCK (MUST BE FIRST)
############################################################
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "0"
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


############################################################
# IMPORTS
############################################################
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

############################################################
# PAGE CONFIG
############################################################
st.set_page_config(page_title="LSTM Trading Dashboard", layout="wide")

st.markdown("""
<style>
body { background-color: #0a0a0a; color: white; }
[data-testid="stAppViewContainer"] { background-color: #0a0a0a; color: #f5f5f5; }
[data-testid="stSidebar"] { background-color: #111111; color: white; }
h1, h2, h3, h4 { color: #00eaff !important; }
</style>
""", unsafe_allow_html=True)

############################################################
# SIMPLE INDICATORS (NO PANDAS_TA)
############################################################
def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def load_data(symbol):
    df = yf.download(symbol, start="2010-01-01", end="2024-12-31")

    if df.empty:
        st.error("No data found.")
        return None

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    df["RSI"] = compute_rsi(df["Close"])

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df.dropna(inplace=True)
    return df


def scale_data(df, features):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler


def create_sequences(data, seq=60):
    X, y = [], []
    for i in range(seq, len(data)):
        X.append(data[i-seq:i])
        y.append(data[i][0])
    return np.array(X), np.array(y)


def backtest(prices, signals, capital=100000):
    cash = capital
    shares = 0
    curve = []
    for price, sig in zip(prices, signals):
        if sig == 1 and cash > 0:
            shares = cash / price
            cash = 0
        elif sig == -1 and shares > 0:
            cash = shares * price
            shares = 0
        curve.append(cash + shares * price)
    return curve


def forecast(model, scaled, scaler, days=30):
    last_seq = scaled[-60:]
    preds_scaled = []
    for _ in range(days):
        x = last_seq.reshape(1, 60, scaled.shape[1])
        pred = model.predict(x, verbose=0)[0][0]
        new_row = np.zeros((scaled.shape[1],))
        new_row[0] = pred
        preds_scaled.append(new_row)
        last_seq = np.vstack([last_seq[1:], new_row])
    preds = scaler.inverse_transform(preds_scaled)[:, 0]
    return preds

############################################################
# SIDEBAR
############################################################
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
days = st.sidebar.slider("Forecast Days", 5, 60, 30)

############################################################
# LOAD DATA
############################################################
df = load_data(symbol)
if df is None:
    st.stop()

features = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal"]
scaled, scaler = scale_data(df, features)

X, y = create_sequences(scaled)
split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

############################################################
# LOAD MODEL (SAFE LOAD)
############################################################
model = load_model("models/lstm_saved_model.keras", compile=False)

############################################################
# PREDICT
############################################################
pred_scaled = model.predict(X_test, verbose=0)

y_test_real = scaler.inverse_transform(
    np.hstack([y_test.reshape(-1,1), np.zeros((len(y_test),5))])
)[:,0]

y_pred_real = scaler.inverse_transform(
    np.hstack([pred_scaled, np.zeros((len(pred_scaled),5))])
)[:,0]

signals = np.where(y_pred_real[1:] > y_test_real[:-1], 1, -1)
signals = np.insert(signals, 0, 0)

portfolio_curve = backtest(y_test_real, signals)
roi = ((portfolio_curve[-1] - 100000)/100000) * 100

############################################################
# TABS
############################################################
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "🤖 Predictions",
    "📈 Backtesting",
    "📥 Reports",
    "📧 Alerts",
    "🆚 Compare Stocks"
])

############################################################
# TAB 1 — Dashboard
############################################################
with tab1:
    st.header(f"📊 Dashboard — {symbol}")

    # Candlestick Chart
    cand = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )])
    cand.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(cand, use_container_width=True)

    # Indicator Chart
    ind = go.Figure()
    ind.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    ind.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))
    ind.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    ind.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.2)
    ind.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.2)
    st.plotly_chart(ind, use_container_width=True)



############################################################
# TAB 2 — Predictions
############################################################
with tab2:
    st.header("🤖 Model Performance")
    st.line_chart(pd.DataFrame({"Actual": y_test_real, "Predicted": y_pred_real}))

    st.subheader("🔮 Future Forecast")
    f = forecast(model, scaled, scaler, days)
    st.line_chart(f)

############################################################
# TAB 3 — Backtesting
############################################################
with tab3:
    st.header("📈 Portfolio Curve")
    st.line_chart(portfolio_curve)
    st.subheader(f"ROI: {roi:.2f}%")

############################################################
# TAB 4 — Reports
############################################################
with tab4:
    st.info("PDF/CSV export coming soon.")

############################################################
# TAB 5 — Alerts
############################################################
with tab5:
    st.info("Email/Telegram trading alerts coming soon.")

############################################################
# TAB 6 — Compare Stocks
############################################################
with tab6:
    st.header("🆚 Multi-Stock Comparison")
    stocks = st.text_input("Enter symbols", "AAPL, TSLA, MSFT")
    syms = [s.strip().upper() for s in stocks.split(",")]
    dfc = pd.DataFrame()
    for s in syms:
        try:
            dfc[s] = yf.download(s, period="1y")["Close"]
        except:
            pass
    if not dfc.empty:
        st.line_chart(dfc)

############################################################

st.success("🚀 Dashboard Ready!")
