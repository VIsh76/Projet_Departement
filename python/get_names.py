from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
import read_file as rf
import numpy as np
import argparse
import os, sys
import pickle

data_path = "../Data_M[2005-2017].csv"
label_path = "../Dico_M[2005-2017].csv"

def seperate_dataset(data_x, data_y, train_ratio):

  assert (train_ratio < 1 and train_ratio > 0), "The train ratio should be between 0 and 1."
  data_num = data_x.shape[0]
  sep_ind = int(data_num * train_ratio)
  return data_x[:sep_ind - 2, :], data_y[2:sep_ind, :], data_x[sep_ind - 2:-5, :], data_y[sep_ind:-3, :]


def names(X, dico_path = "label_path"):
    dico_data = pd.read_csv(dico_path)
    labels, y_name = read_label(dico_path)
    print(labels, y_name)
    for var in X[0]:
        for label in dico_data[2]:
            if(var==label):
                print(var)
def select_seuil_lasso(clf):
    vec_id = []
    for i in range(len(clf.coef_)):
        if(abs(clf.coef_[i])>0.001):
            vec_id.append(clf.get_support(True)[i])
    return np.array(vec_id)

def test():
    if os.path.exists("../Data/datas.npz") and os.path.exists("../Data/datas.pkl"):
      # load datas
      with np.load('../Data/datas.npz') as obj:
        datas = obj['datas']
        dates = obj['dates']

        with open('../Data/datas.pkl', 'rb') as infile:
          indexs = pickle.load(infile)
          labels = pickle.load(infile)
          y_name = pickle.load(infile)

    else:
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
    print("val  : %d label  " % y_data.shape[1])
    print("\n----------------------------------------------")
    print("train: %d feature" % x_data.shape[1])

    # seperate train and validation dataset
    print("\nseperate data...\n")
    x_train, y_train, x_val, y_val = seperate_dataset(x_data, y_data, 0.8)
    print("train: %d cases" % x_train.shape[0])
    print("val  : %d cases" % x_val.shape[0])
    print("\n----------------------------------------------")
    k=20
    feature_eng = SelectKBest(mutual_info_regression, k)
    x_train_new = feature_eng.fit_transform(x_train, y_train[:,0])
    x_val_new = feature_eng.transform(x_val)
    print("keep %d feature" % k)
    print("\n----------------------------------------------")
    feat_selected = feature_eng.get_support(True)
    print("-----------")
    for i in range(len(feat_selected)):
        print(indexs.inv[feat_selected[i]])
    print("\n----------------------------------------------")
    print("train model...\n")
    # print(labels)
    a = labels.set_index('Unnamed: 0')['def'].to_dict();
    for i in feat_selected:
        print(a[i])

test()
