**Stock Price Prediction & Trading Dashboard (LSTM + Streamlit)**

A full-fledged AI-driven stock market analysis platform built using LSTM, technical indicators, backtesting, and a powerful Streamlit UI.
It predicts stock prices, generates buy/sell signals, performs backtesting, and much more.
🔗 Live App: Add after deployment
🔗 Training Notebook: Add your Colab link here

<p align="center">
  <img src="Downloads/dashboard.png" width="800">
</p>

🌟 Features
1. LSTM Stock Price Prediction
<img width="2927" height="1617" alt="7F4298C3-2414-499A-AFF2-624E1B79B2BE" src="https://github.com/user-attachments/assets/24120576-4fcd-45c2-9b26-9bf4454d694b" />


. Predicts next-day price using a deep learning LSTM model


. Uses 60 past days + technical indicators


. Forecasts 5–60 future days


📈 2. Technical Indicators


. SMA 20


. SMA 50


. RSI


. MACD


. Signal Line


🕯️ 3. Candlestick Charts (Plotly)
Interactive TradingView-style charts:


. OHLC candles


. Zooming


. MACD & RSI shaded zones


📊 4. Advanced Backtesting


. Buy/Sell based on model predictions


. Uses capital allocation logic


. Final portfolio value


. ROI %


. Risk management calculator


🔮 5. 30-Day Future Forecasting
Smooth recursive predictions with auto-regression.


💰 6. Crypto Dashboard
Live tracking: BTC, ETH, DOGE (7-day/1h interval)


🆚 7. Multi-Stock Comparison
Compare 2–10 stocks together.


⚙️ 8. Settings


. Auto-refresh


. Symbol selector


. Days selector


. Dark theme (TradingView style)


🤖 9. AI Buy/Sell Recommendation
Based on:


. Trend


. RSI


. MACD–Signal crossover


. Last 5-day movement



📂 Project Structure
StockPriceLSTM/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│     └── lstm_saved_model.keras
│
├── notebooks/
│     └── lstm_training.ipynb
│
├── utils/
│     ├── preprocessing.py
│     └── indicators.py
│
├── images/
│     ├── dashboard.png
│     ├── forecast.png
│     ├── candlestick.png
│     ├── comparison.png
│     ├── crypto.png
│     └── backtest.png
│
└── architecture/
      └── pipeline_diagram.png


🧠 Model Architecture
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                     │ (None, 60, 50)         │        11,400 │
│ dropout (Dropout)               │ (None, 60, 50)         │             0 │
│ lstm_1 (LSTM)                   │ (None, 50)             │        20,200 │
│ dropout_1 (Dropout)             │ (None, 50)             │             0 │
│ dense (Dense)                   │ (None, 25)             │         1,275 │
│ dense_1 (Dense)                 │ (None, 1)              │            26 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 32,901
 Trainable params: 32,901


🔧 Training Details
Dataset: Yahoo Finance (2010–2024)
Sequence Length: 60 days
Optimizer: Adam
Loss Function: MSE
Epochs: 50
Batch Size: 32
Validation Split: 20%
📉 Training Logs (Sample)
Epoch 1/50 — loss: 0.0014 — val_loss: 0.0010
Epoch 8/50 — loss: 0.00039 — val_loss: 0.00080
Epoch 12/50 — loss: 0.000083 — val_loss: 0.00060
Epoch 20/50 — loss: 0.000077 — val_loss: 0.00026
Epoch 33/50 — loss: 0.000062 — val_loss: 0.00038
...

(Add Epoch 34–50 here if you want!)

📊 Model Performance
MetricValueRMSE11.40MAE8.74R² Score0.852 ✔️ Excellent
Interpretation:

Your model explains 85.2% of stock price movement, which is excellent for time-series forecasting.


💹 Backtesting Results
ItemValueInitial Capital₹100,000Final Portfolio Value₹146,359Profit₹46,359ROI46.35%
→ Your LSTM model beats buy-and-hold strategy!

🔮 30-Day Forecast Example
![Forecast](images/forecast.png)


🔧 How to Run Locally
1️⃣ Clone Repo:
git clone https://github.com/aarohi1822/StockPriceLSTM.git
cd StockPriceLSTM

2️⃣ Install Dependencies:
pip install -r requirements.txt

3️⃣ Run Streamlit App:
streamlit run app.py


🔗 Data Source


Yahoo Finance API via yfinance



🧩 Future Improvements


Add transformer/GRU model


Add news sentiment analysis


Include volume/OBV indicator


Add risk-adjusted metrics


Add Telegram alerts


Add Reinforcement Learning trader



Author
Aarohi Gaurav Sharma
B.Tech CSE
AI & ML Developer
GitHub: aarohi1822

🎉 Thank You!
If you like the project, ⭐ the repository!

