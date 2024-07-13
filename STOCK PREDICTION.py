#!/usr/bin/env python
# coding: utf-8

# In[ ]:


TASK 2: STOCK PREDICTION

Founder: Swati Srivastava

Domain: Data Science

Name: Bobade Adedamola Timilehin

Student ID: CA/JN1/13239


# IMPORTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm


# IMPORTING DATA

# In[2]:


df = pd.read_csv('all_stocks_5yr.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Display the first few rows of the dataset
df.head()


# PREPROCESS DATA

# In[4]:


# Sorting byv date
df = df.sort_index()

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']])

lookback = 60

train_data = []
for i in range(lookback, len(scaled_data)):
    train_data.append(scaled_data[i-lookback:i, 0])

train_data = np.array(train_data)
x_train = train_data[:, :-1]
y_train = train_data[:, -1]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# FITTING MODEL AND FORCASTING

# In[ ]:


# Fit the ARIMA model
model = sm.tsa.ARIMA(df['close'], order=(5,1,0))
results = model.fit()

forcast_step=50
forecast = results.forecast(steps=forcast_steps)[0]

# Scale back the forecast to original values
forecast = scaler.inverse_transform(forecast.reshape(-1, 1))


# In[ ]:


FITTING THE


# PLOT AND VISUAL

# In[ ]:


# Generate dates for the forecast
forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps+1, closed='right')

# Plotting the actual stock price
plt.figure(figsize=(14, 5))
plt.plot(df['close'], color='blue', label='Actual Stock Price')
plt.title('Actual Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(loc='upper left')
plt.show()

# Plotting the actual vs predicted stock price
plt.figure(figsize=(14, 5))
plt.plot(df['close'], color='blue', label='Actual Stock Price')
plt.plot(forecast_dates, forecast, color='red', label='Predicted Stock Price')
plt.title('Actual vs Predicted Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(loc='upper left')
plt.show()

# Splitting the data into train and test sets for visualization
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Fit the ARIMA model on the train set
model = sm.tsa.ARIMA(train['close'], order=(5,1,0))
results = model.fit()

# Forecast on the test set
forecast = results.forecast(steps=len(test))[0]

# Scale back the forecast to original values
forecast = scaler.inverse_transform(forecast.reshape(-1, 1))

# Plot the train vs test and predicted stock price
plt.figure(figsize=(14, 5))
plt.plot(train['close'], color='blue', label='Train Stock Price')
plt.plot(test['close'], color='green', label='Test Stock Price')
plt.plot(test.index, forecast, color='red', label='Predicted Stock Price')
plt.title('Train vs Test vs Predicted Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(loc='upper left')
plt.show()

# Plotting the residuals
residuals = results.resid
plt.figure(figsize=(14, 5))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of the ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend(loc='upper left')
plt.show()


# In[ ]:




