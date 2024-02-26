import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load data
quet = pd.read_csv("quet.csv", skiprows=1)

# View data
print(quet.head())

# Plot data to visualise time series
plt.figure()
quet.plot(x='month', y='dispensings', kind='line')
plt.axvline(pd.Timestamp('2014-01-01'), color='lightgray', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Dispensings')
plt.show()

# View ACF/PACF plots of undifferenced data
plot_acf(quet['dispensings'], lags=24)
plt.show()
plot_pacf(quet['dispensings'], lags=24)
plt.show()

# View ACF/PACF plots of differenced/seasonally differenced data
plot_acf(quet['dispensings'].diff().dropna(), lags=24)
plt.show()
plot_pacf(quet['dispensings'].diff().dropna(), lags=24)
plt.show()

# Create variable representing step change and ramp and view
quet['step'] = (quet['month'] >= '2014-01-01').astype(int)
quet['ramp'] = (quet['month'] - pd.Timestamp('2013-12-01')).dt.days // 30
quet.loc[quet['month'] < '2014-01-01', 'ramp'] = 0
print(quet.head())

# Specify first difference = 1 and seasonal difference = 1
# Run model and check residuals
model = ARIMA(quet['dispensings'], order=(2,1,12), seasonal_order=(0,1,1,12))
results = model.fit()
print(results.summary())

# Get confidence intervals
ci = results.conf_int(alpha=0.05)
print(ci)

# To forecast the counterfactual, model data excluding post-intervention time period
quet2 = quet.copy()
quet2.loc[quet2['month'] >= '2014-01-01', 'dispensings'] = np.nan

# Forecast 12 months post-intervention
model2 = ARIMA(quet2['dispensings'], order=(2,1,12), seasonal_order=(0,1,1,12))
results2 = model2.fit()
forecast = results2.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

# Combine with observed data
quet3 = quet.merge(forecast.predicted_mean.rename('forecast'), left_index=True, right_index=True, how='outer')
quet3 = quet3.merge(forecast_ci, left_index=True, right_index=True, how='outer')

# Plot
plt.figure()
plt.plot(quet3['month'], quet3['dispensings'], label='Observed', color='blue')
plt.plot(quet3['month'], quet3['forecast'], label='Forecast', color='red')
plt.fill_between(quet3['month'], quet3['lower dispensings'], quet3['upper dispensings'], color='pink', alpha=0.3)
plt.axvline(pd.Timestamp('2014-01-01'), color='lightgray', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Dispensings')
plt.legend()
plt.show()