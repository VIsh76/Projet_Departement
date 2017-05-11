from sklearn.linear_model import Ridge, Lasso, ElasticNetCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import read_file as rf
import numpy as np
import os, sys
import cPickle as pickle
from bidict import bidict
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

data_path = "../Data_M[2005-2017].csv"
label_path = "../Dico_M[2005-2017].csv"

def seperate_dataset(data_x, data_y, quantity, time):
    """ return x_train, y_train, x_val, y_val """
    data_num = data_x.shape[0]
    return data_x[:quantity - time, :], data_y[time:quantity, 1], data_x[quantity - time: data_num-time, :], data_y[quantity:, 1]


def mae(y, yhat):
  y = y.reshape((1, -1))
  yhat = yhat.reshape((1, -1))
  assert (y.shape == yhat.shape), "y and yhat should have same size"
  return np.mean(np.abs(yhat - y))

#----------------- Load data-----------------#

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

with open('../Data/kmeans.pkl', 'rb') as infile:
  tf_class = pickle.load(infile)

print("load %d data" % datas.shape[0])

# seperate feature(x_train) and prediction(y_train)
y_index = [indexs[i] for i in y_name]
y_data = datas[:, y_index]
x_data = np.delete(datas, y_index, axis = 1)
print("train: %d feature" % x_data.shape[1])
print("val  : %d label  " % y_data.shape[1])
print("\n----------------------------------------------")

#-------------- classify features by name -----------------#

print("\nclustering by name...\n")
tf_index = []
for i in range(len(indexs) - len(y_index)):
  if np.sum(y_index == i) == 0:
      tf_index.append(tf_class[int(indexs.inv[i])])

tf_index = np.array(tf_index)

#----------------- Linear regression ----------------------#

quantity = int(datas.shape[0] * 0.7)
val_num = datas.shape[0] - quantity
month_num = 3
train_results = np.zeros((month_num + 2, quantity - month_num))
valid_results = np.zeros((month_num + 2, val_num))

for i in range(month_num + 1):
    print("\nseperate data for time %i...\n" % i)
    x_train, y_train, x_val, y_val = seperate_dataset(x_data, y_data, quantity, i)
    print x_train.shape
    print y_train.shape
    print x_val.shape
    print y_val.shape
    print("train: %d cases" % x_train.shape[0])
    print("val  : %d cases" % x_val.shape[0])
    print("\n----------------------------------------------")
    

    feature_idx = []
    print("choose data...\n")
    k = 50
    feature_eng = SelectKBest(mutual_info_regression, 1)
    for j in range(k):
        if np.sum(tf_index == j) > 0:
            feature_eng.fit(x_train[:, tf_index == j], y_train)
            feature_idx.append(np.where(tf_index == j)[0][np.argmax(feature_eng.scores_)])

    feature_idx = np.array(feature_idx)
    x_train_new = x_train[:, feature_idx]
    x_val_new = x_val[:, feature_idx]

    print("keep %d feature" % k)
    print("\n----------------------------------------------")

    print("\ntrain elastique net model...\n")
    clf = ElasticNetCV(l1_ratio = [.5, .7, .9, .95, .99, 1], max_iter = 100000)
    clf.fit(x_train_new, y_train)
    yhat_train = clf.predict(x_train_new)
    yhat_val = clf.predict(x_val_new)

    print("score on train: %f" % mae(yhat_train, y_train))
    print("score on val: %f" % mae(yhat_val, y_val))
    print('l1_ratio = %f' % clf.l1_ratio_)
    print('alpha = %f' % clf.alpha_) 

    train_results[i, :] = (yhat_train[month_num - i:]).reshape((1,-1))
    valid_results[i, :] = yhat_val.reshape((1,-1))

#---------------------- time series model --------------#

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

y_log = np.log(y_data[:, 1].astype(float))
y_val_log = np.log(y_data[quantity:,1].astype(float))
history = [x for x in y_log[:month_num]]
time_results = []

for t in range(y_log.shape[0] - month_num):
    try:
        model = ARIMA(history, order = (2, 1, 2))
        model_fit = model.fit(disp = 1)
        output = model_fit.forecast()
        time_results.append(output[0][0])
     
    except:
        time_results.append(np.mean(history[-5:]))
        #print np.mean(history)

    if np.isnan(time_results[t]):
        time_results[t] = np.mean(history[-5:])

    history.append(y_log[month_num + t])

print mae(np.exp(y_log[month_num:]), np.exp(np.array(time_results)))
print np.exp(np.array(time_results)).shape
train_results[month_num+1, :] = np.exp(np.array(time_results[:quantity-month_num])).reshape((1, -1))
valid_results[month_num+1, :] = np.exp(np.array(time_results[quantity-month_num:])).reshape((1, -1))

#------------------ embedding ----------------------#

train_results = train_results[1:]
valid_results = valid_results[1:]

train_labels = y_data[month_num:quantity, 1]
valid_labels = y_data[quantity:, 1]

#print("\ntrain elastique net model for embedding models...\n")
#clf = ElasticNetCV(l1_ratio = [.5, .7, .9, .95, .99, 1], max_iter = 100000)
#clf.fit(train_results, train_labels)
#yhat_train = clf.predict(train_results)
#yhat_val = clf.predict(valid_results)

yhat_train = train_results.mean(0)
yhat_val = valid_results.mean(0)

print("score on train: %f" % mae(yhat_train, train_labels))
print("score on val: %f" % mae(yhat_val, valid_labels))
print('l1_ratio = %f' % clf.l1_ratio_)
print('alpha = %f' % clf.alpha_) 
