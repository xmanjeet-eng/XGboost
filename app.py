import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from flask import Flask, render_template
from xgboost import XGBRegressor
from datetime import datetime, timedelta

app = Flask(__name__)

def get_intraday_data(ticker):
    # Fetch 5-minute interval data for precise intraday patterns
    df = yf.download(ticker, period='5d', interval='5m', multi_level_index=False)
    df.columns = [str(col) for col in df.columns]
    
    # Advanced Alpha Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Create Target: Next 15-min move
    df['Target'] = df['Close'].shift(-3) 
    df.dropna(inplace=True)
    
    features = ['Close', 'RSI', 'MACD', 'VWAP', 'EMA_9', 'ATR']
    X = df[features][:-3]
    y = df['Target'][:-3]
    
    # XGBoost for high-speed prediction
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(X, y)
    
    last_row = df[features].tail(1)
    prediction = model.predict(last_row)[0]
    current = df['Close'].iloc[-1]
    
    # Logic for Signals
    diff = ((prediction - current) / current) * 100
    signal = "NEUTRAL"
    if diff > 0.15: signal = "STRONG BUY"
    elif diff < -0.15: signal = "STRONG SELL"
    
    return {
        "symbol": "NIFTY 50" if "NSEI" in ticker else "BANK NIFTY",
        "current": round(current, 2),
        "target": round(prediction, 2),
        "signal": signal,
        "rsi": int(df['RSI'].iloc[-1]),
        "vwap_status": "ABOVE" if current > df['VWAP'].iloc[-1] else "BELOW",
        "change": round(diff, 2)
    }

@app.route('/')
def home():
    try:
        nifty = get_intraday_data('^NSEI')
        bnifty = get_intraday_data('^NSEBANK')
        return render_template('index.html', n=nifty, b=bnifty, time=datetime.now().strftime('%H:%M:%S'))
    except Exception as e:
        return f"System Error: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
