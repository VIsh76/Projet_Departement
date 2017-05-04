import read_file as rf
import numpy as np
import os, sys
import cPickle as pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

data_path = "../Data_M[2005-2017].csv"
label_path = "../Dico_M[2005-2017].csv"


def seperate_dataset(data_y, train_ratio):
  assert (train_ratio < 1 and train_ratio > 0), "The train ratio should be between 0 and 1."
  data_num = data_y.shape[0]
  sep_ind = int(data_num * train_ratio)
  return data_y[:sep_ind, :], data_y[sep_ind:-3, :]


def mae(y, yhat):
  y = y.reshape((1, -1))
  yhat = yhat.reshape((1, -1))
  assert (y.shape == yhat.shape), "y and yhat should have same size"
  return np.mean(np.abs(yhat - y))



def mse(y, yhat):
  y = y.reshape((1, -1))
  yhat = yhat.reshape((1, -1))
  assert (y.shape == yhat.shape), "y and yhat should have same size"
  return np.mean(np.power((yhat - y), 2))


print("\nload data: " + data_path.split('/')[-1].split('.')[0])

if os.path.exists("../Data/datas.npz") and os.path.exists("../Data/datas.pkl"):
  print "0"
  # load datas
  with np.load('../Data/datas.npz') as obj:
    datas = obj['datas']
    dates = obj['dates']

  with open('../Data/datas.pkl', 'rb') as infile:
    indexs = pickle.load(infile)
    labels = pickle.load(infile)
    y_name = pickle.load(infile)

else:
  print "1"
  # load datas
  labels, y_name = rf.read_label(label_path)
  dates, datas, indexs = rf.read_data(data_path, y_name, 0)

  # save datas
  np.savez('../Data/datas.npz', datas=datas, dates=dates)
  with open('../Data/datas.pkl', 'wb') as outfile:
    pickle.dump(indexs, outfile)
    pickle.dump(labels, outfile)
    pickle.dump(y_name, outfile)

print("load %d data" % datas.shape[0])

# seperate feature(x_train) and prediction(y_train)
y_index = [indexs[i] for i in y_name]
y_data = datas[:, y_index]
x_data = np.delete(datas, y_index, axis = 1)
print("train: %d feature" % x_data.shape[1])
print("val  : %d label  " % y_data.shape[1])
print("\n----------------------------------------------")

# seperate train and validation dataset
print("\nseperate data...\n")
y_train, y_val = seperate_dataset(y_data, 0.8)
print("train: %d cases" % y_train.shape[0])
print("val  : %d cases" % y_val.shape[0])
print("\n----------------------------------------------")


def moving_average(a, n=3) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

def moving_std(a, b, n=3):
  ret = [0 for i in range(a.shape[0])]

  for i in range(len(ret)):
    if i == 0 or i == 1:
      continue
    else:
      ret[i] = np.sqrt(np.mean(np.power(a[i-n+1:i+1] - b[i-n+1], 2)))

  return np.array(ret[n-1:])

def test_stationarity(train_data):
  # calculate moving_average and moving_std
  rolmean = moving_average(train_data, 12)
  rolstd = moving_std(train_data, rolmean, 12)

  plt.figure(1, figsize=(8, 5))
  plt.plot(train_data[2:], color='blue', label='Data')
  plt.plot(rolmean, color='red', label='rolling mean')
  plt.plot(rolstd, color='black', label='rolling std')
  plt.legend(loc='best')
  plt.title('Rolling mean & standard deviation')
  plt.show()

  # Perform Dickey-Fuller test
  print('Results of Dickey-Fuller Test:')
  dftest = adfuller(train_data, autolag='AIC')
  dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
  for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

  print(dfoutput)

y_log = np.log(y_train[:, 1].astype(float))
"""
test_stationarity(y_train[:, 1])
y_avg = moving_average(y_log, 12)
y_log_avg_diff = y_log[11:] - y_avg
test_stationarity(y_log_avg_diff)

# Eliminating Trend and Seasonality
y_log_diff = y_log[1:] - y_log[:-1]
test_stationarity(y_log_diff)

# Decomposing

decomposition = seasonal_decompose(y_log, freq=12)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(4, figsize=(8, 5))
plt.subplot(411)
plt.plot(y_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

residual = residual[~np.isnan(residual)]
test_stationarity(residual)

# ACF and PACF plots

lag_acf = acf(y_log_diff, nlags = 20)
lag_pacf = pacf(y_log_diff, nlags = 20, method='ols')


# ACF
# q = 2
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(y_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(y_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

# Plot PACF
# p = 3
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(y_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(y_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()
"""

"""
# ARIMA model
y_val_log =  np.log(y_val[:, 1].astype(float))
history = [x for x in y_log]
predictions = []

for t in range(y_val.shape[0]):
  try:
    model = ARIMA(history, order = (2, 1, 2))
    model_fit = model.fit(disp = 1)
    output = model_fit.forecast()
    predictions.append(output[0][0])

    
  except:
    predictions.append(np.mean(history[-5:]))
    #print np.mean(history)

  if np.isnan(predictions[t]):
    predictions[t] = np.mean(history[-5:])

  history.append(y_val_log[t])

print np.exp(y_val_log)
print np.exp(np.array(predictions))
print mae(np.exp(y_val_log), np.exp(np.array(predictions)))
"""
# ARIMA
y_val_log =  np.log(y_data[:-3, 1].astype(float))
history = [x for x in y_val_log[:3]]
predictions = []

for t in range(y_val_log.shape[0] - 3):
  try:
    model = ARIMA(history, order = (2, 1, 2))
    model_fit = model.fit(disp = 1)
    output = model_fit.forecast()
    predictions.append(output[0][0])

    
  except:
    predictions.append(np.mean(history[-5:]))
    #print np.mean(history)

  if np.isnan(predictions[t]):
    predictions[t] = np.mean(history[-5:])

  history.append(y_val_log[t+3])

print np.exp(y_val_log).shape
print np.exp(np.array(predictions)).shape
print mae(np.exp(y_val_log)[3:], np.exp(np.array(predictions)))


# SARIMAX
"""
y_val_log =  np.log(y_val[:, 1].astype(float))
history = [x for x in y_log]
predictions = []

for t in range(y_val_log.shape[0]):
  try:
    model = SARIMAX(history, trend = 'n', order = (2, 1, 2), seasonal_order=(0,1,1,12))
    model_fit = model.fit()
    output = model_fit.forecast()
    predictions.append(output[0][0])

    
  except:
    predictions.append(np.mean(history[-5:]))
    #print np.mean(history)

  if np.isnan(predictions[t]):
    predictions[t] = np.mean(history[-5:])

  history.append(y_val_log[t])

print mae(np.exp(y_val_log), np.exp(np.array(predictions)))
"""
