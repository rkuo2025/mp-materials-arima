import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import datetime

def build_mp_arima():
    ticker = "MP"
    print(f"--- Predicting Price for {ticker} (MP Materials) ---")
    
    # 1. Data Collection
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365*2) # 2 years of data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("Failed to download data.")
        return
        
    series = data['Close'].dropna()
    
    # 2. Stationarity Check (Augmented Dickey-Fuller test)
    # Most stock data is non-stationary (d=1)
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    
    # 3. Differencing (d parameter)
    # If p-value > 0.05, we difference the data
    d = 1 if result[1] > 0.05 else 0
    print(f"Selected differencing (d): {d}")
    
    # 4. Model Selection (Tuning p, q)
    # Standard research often uses ACF/PACF or AIC minimization.
    # For this script, we'll use a common robust order (5, d, 0) as a baseline.
    p, q = 5, 0 
    
    # 5. Build and Fit the Model
    print(f"Fitting ARIMA({p}, {d}, {q}) model...")
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
    
    # 6. Forecasting (Next 30 days)
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()
    
    # Create dates for forecast
    last_date = series.index[-1]
    forecast_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=forecast_steps)
    forecast_df.index = forecast_dates
    
    # 7. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(series.tail(100), label='Historical (Last 100 days)')
    plt.plot(forecast_df['mean'], color='red', label='Forecast')
    plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3, label='95% CI')
    plt.title(f'MP Materials (MP) Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('mp_prediction_plot.png')
    
    print("\nForecast for next 5 days:")
    print(forecast_df['mean'].head(5))
    print(f"\nPlot saved to mp_prediction_plot.png")

if __name__ == "__main__":
    build_mp_arima()
