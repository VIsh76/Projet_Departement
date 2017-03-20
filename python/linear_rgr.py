from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
import read_file as rf
import numpy as np

data_path = "../Data_M[2005-2017].csv"
label_path = "../Dico_M[2005-2017].csv"

def seperate_dataset(data_x, data_y, train_ratio):

  assert (train_ratio < 1 and train_ratio > 0), "The train ratio should be between 0 and 1."
  data_num = data_x.shape[0]
  sep_ind = int(data_num * train_ratio)
  return data_x[:sep_ind - 2, :], data_y[2:sep_ind, :], data_x[sep_ind - 2:-5, :], data_y[sep_ind:-3, :]

def mse(y, yhat):

  y = y.reshape((1, -1))
  yhat = yhat.reshape((1, -1))
  assert (y.shape == yhat.shape), "y and yhat should have same size"
  return np.sum(np.power((yhat - y), 2)) / y.shape[-1]

print "\nload data: " + data_path.split('/')[-1].split('.')[0]
labels, y_name = rf.read_label(label_path)
dates, datas, indexs = rf.read_data(data_path, y_name, 0.3)
print "load %d data" % datas.shape[0]

# seperate feature(x_train) and prediction(y_train)
y_index = [indexs[i] for i in y_name]
y_data = datas[:, y_index]
x_data = np.delete(datas, y_index, axis = 1)
print "train: %d feature" % x_data.shape[1]
print "val  : %d label  " % y_data.shape[1]
print "\n----------------------------------------------"

# seperate train and validation dataset
print "\nseperate data...\n"
x_train, y_train, x_val, y_val = seperate_dataset(x_data, y_data, 0.8)
print "train: %d cases" % x_train.shape[0]
print "val  : %d cases" % x_val.shape[0]
print "\n----------------------------------------------"

print "choose data...\n"
k = 100
feature_eng = SelectKBest(mutual_info_regression, k)
x_train_new = feature_eng.fit_transform(x_train, y_train[:,0])
x_val_new = feature_eng.transform(x_val)
print "keep %d feature" % k
print "\n----------------------------------------------"

print "train model...\n"
clf = Lasso(alpha=0.1)
clf.fit(x_train_new, y_train[:, 0])
yhat_train = clf.predict(x_train_new)
yhat_val = clf.predict(x_val_new)
print "score on train: %f" % mse(yhat_train, y_train[:, 0])
print "score on val: %f" % mse(yhat_val, y_val[:, 0]) 
