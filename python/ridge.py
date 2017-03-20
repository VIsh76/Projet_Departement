from sklearn.linear_model import Ridge
import read_file as rf
import numpy as np

data_path = "../Data_M.csv"
label_path = "../Dico_M.csv"

def seperate_dataset(data_x, data_y, train_ratio):

  assert (train_ratio < 1 and train_ratio > 0), "The train ratio should be between 0 and 1."
  data_num = data_x.shape[0]
  sep_ind = int(data_num * train_ratio)

  return data_x[:sep_ind - 2, :], data_y[2:sep_ind, :], data_x[sep_ind - 2:-2, :], data_y[sep_ind:, :]



print "load data...\n"
labels, y_name = rf.read_label(label_path)
dates, datas, indexs = rf.read_data(data_path, 0.3)

# seperate feature(x_train) and prediction(y_train)
y_index = [indexs[i] for i in y_name]
y_data = datas[:, y_index]
x_data = np.delete(datas, y_index, axis = 1)

# seperate train and validation dataset
print "seperate data...\n"
x_train, y_train, x_val, y_val = seperate_dataset(x_data, y_data, 0.7)
print x_train.shape, y_train.shape, x_val.shape, y_val.shape

print "train model...\n"
clf = Ridge(alpha=1.0)
clf.fit(x_train, y_train[:, 0])
print clf.score(x_val, y_val[:, 0])  
