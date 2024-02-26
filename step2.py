import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

# Load data
quet = pd.read_csv('quet.csv', parse_dates=['date'], index_col='date')

# Convert data to time series
quet_ts = quet['dispensings'].asfreq('MS')

# Plot data
plt.figure(figsize=(10, 6))
quet_ts.plot(color='blue', label='Dispensings')
plt.axvline(pd.Timestamp('2014-01-01'), color='gray', linestyle='--', label='Intervention')
plt.xlabel('Month')
plt.ylabel('Dispensings')
plt.legend()
plt.show()

# ACF and PACF plots
plot_acf(quet_ts, lags=24)
plot_pacf(quet_ts, lags=24)
plt.show()

# Create step and ramp variables
quet['step'] = (quet.index >= '2014-01-01').astype(int)
quet['ramp'] = np.arange(len(quet))
quet.loc[quet.index < '2014-01-01', 'ramp'] = 0

# Fit the model
model = SARIMAX(quet_ts, order=(2, 1, 0), seasonal_order=(0, 1, 1, 12),
                exog=quet[['step', 'ramp']])
results = model.fit(disp=False)

# Check residuals
residuals = results.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.show()

# Forecasting
future_dates = [quet_ts.index[-1] + DateOffset(months=x) for x in range(1, 13)]
forecast = results.get_forecast(steps=12, exog=quet[['step', 'ramp']].iloc[-12:])
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()

# Combine observed and forecasted data
combined = pd.concat([quet_ts, mean_forecast])
combined.plot(figsize=(10, 6))
plt.fill_between(future_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.axvline(pd.Timestamp('2014-01-01'), color='gray', linestyle='--', label='Intervention')
plt.xlabel('Month')
plt.ylabel('Dispensings')
plt.legend(['Observed', 'Forecast'])
plt.show()
