import pandas as pd
import numpy as np
import os, sys
from scipy.stats import mode
from sklearn.decomposition import PCA

data_path = "../Data_M.csv"
label_path = "../Dico_M.csv"


def read_data(fp_path, y_name, ratio):
  '''
  A function to read data files. It will replace NA with mode and normalize datas.
  Input:
    - fp_path : the path of file
    -  ratio  : keep columns where we find more than r% valid numbers
  Output:
    - o1 : the date of each month
    - o2 : the feature of each month
    - o3 : a dictionary showing the relationship between feature name(like '0016456758') and columns number
  '''

  fp_data = pd.read_csv(fp_path)
  cols = np.array(fp_data.apply(num_missing, axis = 0))
  name_cols = np.array(fp_data.columns)

  # delete invalid colomns
  for i in range(cols.shape[0]):
    if cols[i] > fp_data.shape[0] * ratio or fp_data[name_cols[i]].min() == fp_data[name_cols[i]].max():
      if name_cols[i] in y_name:
        pass
      else:
        fp_data = fp_data.drop(name_cols[i], 1)

  # normalize data to [0, 1]
  name_cols = np.array(fp_data.columns)
  for name in name_cols[2:]:
    if name in y_name:
      pass
    else:
      fp_data[name].fillna(mode(fp_data[name]).mode[0], inplace = True)
      fp_data[name] = (fp_data[name] - fp_data[name].min()) / (fp_data[name].max() - fp_data[name].min())

  return np.array(fp_data)[:, 1], np.array(fp_data)[:, 2:], dict(zip(np.array(fp_data.columns)[2:], range(fp_data.shape[1] - 2)))


def num_missing(x):
  return sum(x.isnull())

def read_label(fp_path):
  """
  A function to read label files. It returns feature names which we have to predict.
  Input: fp_path - file path
  Output:
    - fp_label: the whole label file
    - y_name: a list contains the names of features to predict
  """
  fp_label = pd.read_csv(fp_path)
  var_name = np.array(fp_label['def'])
  id_bank = np.array(fp_label['IDBANK'], dtype = np.str)

  # find predict feature
  y_name = [('00' + id_bank[i]) for i in range(len(var_name)) if var_name[i].find('faill') != -1 and var_name[i].find('CVS') != -1]

  return fp_label, y_name

if "__name__" == "__main__":

  labels, y_name = read_label(label_path)
  dates, datas, indexs = read_data(data_path, 0.3)

  # seperate feature(x_train) and prediction(y_train)
  y_index = [indexs[i] for i in y_name]
  y_train = datas[:, y_index]
  x_train = np.delete(datas, y_index, axis = 1)
