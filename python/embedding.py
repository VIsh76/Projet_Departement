from sklearn.linear_model import Ridge, Lasso, ElasticNetCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import read_file as rf
import numpy as np
import os, sys
import cPickle as pickle
from bidict import bidict
import pandas as pd
import itertools
from operator import itemgetter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

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

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

def separate_train(X,y,k):
    score=[[i,0] for i in range(X.shape[1])]
    d=X.shape[0]/10        
    r=X.shape[0]%10
    for i in range(10): 
        if 10>i+r:
            X2=X[(d*i):(d*(i+1)),:]
            y2=y[(d*i):(d*(i+1))] 
        else: 
            X2=X[(d*i+r+i-10):(d*(i+1)+r+(i+1)-10),:]
            y2=y[(d*i+r+i-10):(d*(i+1)+r+(i+1)-10)]
        res=Corr(X2,y2)
        res=sorted(res,key=itemgetter(0),reverse=True)
        res=res[0:k]
        for j in range(k):
            score[res[j][1]][1]+=res[j][0]*np.sqrt(i+1)
    return score


def corr(x,y):
    if(x.std(0)==0):
        return 0
    res=abs((x*y).mean(0)-x.mean(0)*y.mean(0))/(x.std(0)*y.std(0))
    if(res>1):
        return 0
    else:
        return res

def Corr(X,y):
    res=[]
    for i in range(X.shape[1]):
        res.append([corr(X[:,i],y),i])
    return res

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
train_errors = []
test_errors = []
quantity = int(datas.shape[0] * 0.75)
val_num = datas.shape[0] - quantity
month_num = 3
train_results = np.zeros((month_num + 1, quantity - month_num))
valid_results = np.zeros((month_num + 1, val_num))

k = 10
print("\nselect %d features..." % k)

x_train, y_train, x_val, y_val = seperate_dataset(x_data, y_data, quantity, 2)
feature_idx = []
print("choose data...\n")
feature_eng = SelectKBest(mutual_info_regression, 10)
for j in range(k):
    if np.sum(tf_index == j) > 0:
        feature_eng.fit(x_train[:, tf_index == j], y_train)
        feature_idx.append(np.where(tf_index == j)[0][np.argsort(feature_eng.scores_)[::-1][0]])
feature_idx = np.array(feature_idx)


"""
score=separate_train(x_train[:, feature_idx],y_train,k)
score=sorted(score,key=itemgetter(1),reverse=True)
score=[i[0] for i in score]
feature_idx=feature_idx[score[0:k]]
"""


for i in range(1, month_num + 1):
    print("\nseperate data for time %i...\n" % i)
    x_train, y_train, x_val, y_val = seperate_dataset(x_data, y_data, quantity, i)
    print x_train.shape
    print y_train.shape
    print x_val.shape
    print y_val.shape
    print("train: %d cases" % x_train.shape[0])
    print("val  : %d cases" % x_val.shape[0])
    print("\n----------------------------------------------")
    
    x_train_new = x_train[:, feature_idx]
    x_val_new = x_val[:, feature_idx]

    print("keep %d feature" % k)
    print("\n----------------------------------------------")

    print("\ntrain elastique net model...\n")
    clf = ElasticNetCV(l1_ratio = [.5, .7, .9, .95, .99, 1], max_iter = 100000)
    clf.fit(x_train_new, y_train)
    yhat_train = clf.predict(x_train_new)
    yhat_val = clf.predict(x_val_new)

    train_errors.append(mae(yhat_train, y_train))
    test_errors.append(mae(yhat_val, y_val))
    print("score on train: %f" % mae(yhat_train, y_train))
    print("score on val: %f" % mae(yhat_val, y_val))
    print('l1_ratio = %f' % clf.l1_ratio_)
    print('alpha = %f' % clf.alpha_) 

    train_results[i-1, :] = (yhat_train[month_num - i:]).reshape((1,-1))
    valid_results[i-1, :] = yhat_val.reshape((1,-1))

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

test_errors.append(mae(np.exp(y_log[month_num:]), np.exp(np.array(time_results))))
print mae(np.exp(y_log[month_num:]), np.exp(np.array(time_results)))
print np.exp(np.array(time_results)).shape
train_results[month_num, :] = np.exp(np.array(time_results[:quantity-month_num])).reshape((1, -1))
valid_results[month_num, :] = np.exp(np.array(time_results[quantity-month_num:])).reshape((1, -1))

#------------------ embedding ----------------------#

test_errors = np.array(test_errors)
idx = np.where(test_errors < 25)[0]
train_results = train_results[idx, :]
valid_results = valid_results[idx, :]


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
#print('l1_ratio = %f' % clf.l1_ratio_)
#print('alpha = %f' % clf.alpha_) 

plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(valid_labels)), valid_labels, label='ground-truth', color='red')
plt.plot(np.arange(len(valid_labels)), yhat_val, label='prediction', color='blue')
plt.xlabel('t')
plt.ylabel('y')
plt.title('The final model')
plt.legend()
plt.grid()
plt.show()

_, _, dict_name = rf.read_label(label_path)
for u in range(k):
    print(feature_idx[u])
    print(indexs.inv[feature_idx[u]])
    print(dict_name[indexs.inv[feature_idx[u]]])

