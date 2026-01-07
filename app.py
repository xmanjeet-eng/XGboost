import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from flask import Flask, render_template
from xgboost import XGBRegressor
from datetime import datetime

app = Flask(__name__)

def get_alpha_signal(ticker):
    # Fetch 5-minute interval for intraday accuracy (last 5 days)
    df = yf.download(ticker, period='5d', interval='5m', multi_level_index=False)
    if df.empty: return None
    
    # 1. TECHNICAL LAYER (Momentum & Trend)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    
    # 2. VOLATILITY LAYER (Fear Index Fusion)
    vix = yf.download('^INDIAVIX', period='5d', interval='5m', multi_level_index=False)
    df['VIX'] = vix['Close'].reindex(df.index, method='ffill')
    
    # 3. PREDICTION ENGINE (XGBoost)
    # Goal: Predict price 15 minutes into the future
    df['Target'] = df['Close'].shift(-3) 
    features = ['Close', 'RSI', 'VIX', 'EMA_20']
    
    train_df = df.dropna()
    X = train_df[features]
    y = train_df['Target']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5)
    model.fit(X, y)
    
    # Generate Live Output
    latest = df[features].tail(1)
    prediction = model.predict(latest)[0]
    curr = df['Close'].iloc[-1]
    
    # Probability Logic
    diff = ((prediction - curr) / curr) * 100
    prob_up = min(max(50 + (diff * 10), 10), 90) # Weighted probability
    
    return {
        "symbol": "NIFTY 50" if "NSEI" in ticker else "BANK NIFTY",
        "current": f"{curr:,.2f}",
        "predicted": f"{prediction:,.2f}",
        "up_prob": round(prob_up, 1),
        "down_prob": round(100 - prob_up, 1),
        "rsi": int(df['RSI'].iloc[-1]),
        "vix": round(df['VIX'].iloc[-1], 2),
        "signal": "BULLISH" if diff > 0.05 else "BEARISH" if diff < -0.05 else "NEUTRAL"
    }

@app.route('/')
def home():
    try:
        nifty = get_alpha_signal('^NSEI')
        bank = get_alpha_signal('^NSEBANK')
        return render_template('index.html', n=nifty, b=bank, ts=datetime.now().strftime('%H:%M:%S'))
    except Exception as e:
        return f"Terminal Error: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
