from sklearn.linear_model import Ridge, Lasso, ElasticNetCV, LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import read_file as rf
import numpy as np
import argparse
import os, sys
import cPickle as pickle
from bidict import bidict
import matplotlib.pyplot as plt

data_path = "../Data_M[2005-2017].csv"
label_path = "../Dico_M[2005-2017].csv"


def seperate_dataset(data_x, data_y, train_ratio):
  assert (train_ratio < 1 and train_ratio > 0), "The train ratio should be between 0 and 1."
  data_num = data_x.shape[0]
  sep_ind = int(data_num * train_ratio)
  return data_x[:sep_ind - 2, :], data_y[2:sep_ind, :], data_x[sep_ind - 2:-2, :], data_y[sep_ind:, :]


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

# seperate train and validation dataset
print("\nseperate data...\n")
x_train, y_train, x_val, y_val = seperate_dataset(x_data, y_data, 0.7)
print("train: %d cases" % x_train.shape[0])
print("val  : %d cases" % x_val.shape[0])
print("\n----------------------------------------------")

# class features by group
tf_index = []
for i in range(len(indexs) - len(y_index)):
  if np.sum(y_index == i) == 0:
      tf_index.append(tf_class[int(indexs.inv[i])])


tf_index = np.array(tf_index)

# choose features
feature_idx = []
print("choose data...\n")
k = 10
feature_eng = SelectKBest(mutual_info_regression, 1)
for i in range(k):
  if np.sum(tf_index == i) > 0:
    feature_eng.fit(x_train[:, tf_index == i], y_train[:,1])
    feature_idx.append(np.where(tf_index == i)[0][np.argmax(feature_eng.scores_)])

feature_idx = np.array(feature_idx)
x_train_new = x_train[:, feature_idx]
x_val_new = x_val[:, feature_idx]

print("keep %d feature" % k)
print("\n----------------------------------------------")

print("train LASSO model...\n")
clf = Lasso(alpha=0.5)
clf.fit(x_train_new, y_train[:, 1])
yhat_train_lasso = clf.predict(x_train_new)
yhat_val_lasso = clf.predict(x_val_new)
print("score on train: %f" % mae(yhat_train_lasso, y_train[:, 1]))
print("score on val: %f" % mae(yhat_val_lasso, y_val[:, 1]))

print("train ridge model...\n")
clf = Ridge(alpha=0.5)
clf.fit(x_train_new, y_train[:, 1])
yhat_train_ridge = clf.predict(x_train_new)
yhat_val_ridge = clf.predict(x_val_new)
print("score on train: %f" % mae(yhat_train_ridge, y_train[:, 1]))
print("score on val: %f" % mae(yhat_val_ridge, y_val[:, 1]))

print("train LR model...\n")
clf = LinearRegression()
clf.fit(x_train_new, y_train[:, 1])
yhat_train_lr = clf.predict(x_train_new)
yhat_val_lr = clf.predict(x_val_new)
print("score on train: %f" % mae(yhat_train_lr, y_train[:, 1]))
print("score on val: %f" % mae(yhat_val_lr, y_val[:, 1]))

print("\ntrain elastique net model...\n")
clf = ElasticNetCV(l1_ratio = [.5, .7, .9, .95, .99, 1], max_iter = 100000)
clf.fit(x_train_new, y_train[:, 1])
yhat_train_e = clf.predict(x_train_new)
yhat_val_e = clf.predict(x_val_new)
print("score on train: %f" % mae(yhat_train_e, y_train[:, 1]))
print("score on val: %f" % mae(yhat_val_e, y_val[:, 1]))
print('l1_ratio = %f' % clf.l1_ratio_)
print('alpha = %f' % clf.alpha_)

plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(y_val[:, 1])), y_val[:, 1], label='ground-truth', color='red')
plt.plot(np.arange(len(y_val[:, 1])), yhat_val_lasso, label='prediction', color='blue')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Lasso model')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(y_val[:, 1])), y_val[:, 1], label='ground-truth', color='red')
plt.plot(np.arange(len(y_val[:, 1])), yhat_val_ridge, label='prediction', color='blue')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Ridge model')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(y_val[:, 1])), y_val[:, 1], label='ground-truth', color='red')
plt.plot(np.arange(len(y_val[:, 1])), yhat_val_lr, label='prediction', color='blue')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Linear regression model')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(y_val[:, 1])), y_val[:, 1], label='ground-truth', color='red')
plt.plot(np.arange(len(y_val[:, 1])), yhat_val_e, label='prediction', color='blue')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Elastic net')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(y_val[:, 1])), y_val[:, 1], label='ground-truth', color='red')
plt.plot(np.arange(len(y_val[:, 1])), yhat_val_lasso, label='lasso', color='blue')
plt.plot(np.arange(len(y_val[:, 1])), yhat_val_ridge, label='ridge', color='black')
plt.plot(np.arange(len(y_val[:, 1])), yhat_val_lr, label='linear regression', color='yellow')
plt.plot(np.arange(len(y_val[:, 1])), yhat_val_e, label='elastic net', color='pink')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Model comparaison')
plt.legend()
plt.grid()
plt.show()

"""
print("keep %d feature" % k)
print("\n----------------------------------------------")

print("train LASSO model...\n")
clf = Lasso(alpha=1)
clf.fit(x_train_new, y_train[:, 1])
yhat_train = clf.predict(x_train_new)
yhat_val = clf.predict(x_val_new)
print("score on train: %f" % mae(yhat_train, y_train[:, 1]))
print("score on val: %f" % mae(yhat_val, y_val[:, 1]))


print(y_val[:, 1])
print(yhat_val)

print("\ntrain SVR model...\n")
clf = SVR(C=1, epsilon=0.2)
clf.fit(x_train_new, y_train[:, 1])
yhat_train = clf.predict(x_train_new)
yhat_val = clf.predict(x_val_new)
print("score on train: %f" % mae(yhat_train, y_train[:, 1]))
print("score on val: %f" % mae(yhat_val, y_val[:, 1]))


print(y_val[:, 1])
print(yhat_val)


print("\ntrain decision tree model...\n")
clf = DecisionTreeRegressor(max_depth=2)
clf.fit(x_train_new, y_train[:, 1])
yhat_train = clf.predict(x_train_new)
yhat_val = clf.predict(x_val_new)
print("score on train: %f" % mae(yhat_train, y_train[:, 1]))
print("score on val: %f" % mae(yhat_val, y_val[:, 1]))

print(y_val[:, 1])
print(yhat_val)


print("\ntrain elastique net model...\n")
clf = ElasticNetCV(l1_ratio = [.5, .7, .9, .95, .99, 1], max_iter = 100000)
clf.fit(x_train_new, y_train[:, 1])
yhat_train = clf.predict(x_train_new)
yhat_val = clf.predict(x_val_new)
print("score on train: %f" % mae(yhat_train, y_train[:, 1]))
print("score on val: %f" % mae(yhat_val, y_val[:, 1]))
print('l1_ratio = %f' % clf.l1_ratio_)
print('alpha = %f' % clf.alpha_) 

print(y_val[:, 1])
print(yhat_val)"""
