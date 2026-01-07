import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from flask import Flask, render_template
from xgboost import XGBRegressor
from datetime import datetime

app = Flask(__name__)

def get_alpha_signal(ticker):
    # Fetch 5m intraday data (Last 5 days to ensure enough RSI data)
    df = yf.download(ticker, period='5d', interval='5m', multi_level_index=False)
    if df.empty: return None
    
    # Feature Engineering
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    
    # India VIX Fusion
    vix = yf.download('^INDIAVIX', period='5d', interval='5m', multi_level_index=False)
    df['VIX'] = vix['Close'].reindex(df.index, method='ffill')
    
    # Target Prediction (Next 15 min)
    df['Target'] = df['Close'].shift(-3)
    features = ['Close', 'RSI', 'VIX', 'EMA_20']
    
    train = df.dropna()
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08)
    model.fit(train[features], train['Target'])
    
    # Logic
    latest = df[features].tail(1)
    pred = model.predict(latest)[0]
    curr = df['Close'].iloc[-1]
    
    diff = ((pred - curr) / curr) * 100
    prob_up = min(max(50 + (diff * 12), 15), 85)
    
    return {
        "symbol": "NIFTY 50" if "NSEI" in ticker else "BANK NIFTY",
        "current": f"{curr:,.2f}",
        "predicted": f"{pred:,.2f}",
        "up_prob": round(prob_up, 1),
        "down_prob": round(100 - prob_up, 1),
        "rsi": int(df['RSI'].iloc[-1]),
        "vix": round(df['VIX'].iloc[-1], 2),
        "signal": "Strong Bull" if diff > 0.08 else "Strong Bear" if diff < -0.08 else "Neutral Scan"
    }

@app.route('/')
def home():
    try:
        nifty_data = get_alpha_signal('^NSEI')
        bank_data = get_alpha_signal('^NSEBANK')
        return render_template('index.html', n=nifty_data, b=bank_data, ts=datetime.now().strftime('%H:%M:%S'))
    except Exception as e:
        return f"Terminal Offline: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
