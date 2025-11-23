from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_model(data, periods=3):
    """
    Seasonal ARIMA forecast for one time series
    """
    model = SARIMAX(
        data,
        order=(1,1,1),            # ARIMA(p,d,q)
        seasonal_order=(1,1,1,12) # annual seasonality
    )
    results = model.fit(disp=False)
    forecast = results.forecast(periods)
    return forecast

