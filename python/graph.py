from static import *
import read_file as rf
import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
import read_file as rf
import numpy as np
import argparse
import os, sys
import pickle
# labels, y_name = rf.read_label(StatObj.label_path())
# print('-------------------------------')
# print(type(labels))
data_path = "../Data_M.csv"
label_path = "../Dico_M.csv"

def trace(datas,dates,nb=10):
    time = np.arange(1,nb,1000)
    for i in range(nb):
        plt.plot(datas[:,-i])
    plt.title("Hola")
    plt.show()

def trace_y(Y):
    for i in range(Y[0]):
        plt.plot(Y[:,i])
    plt.show()


def read_fast():
    if os.path.exists("../Data/datas.npy"):
      # load datas
      datas = np.load('../Data/datas.npy')
      dates = np.load('../Data/dates.npy')
      outfile = open('../Data/indexs.pkl', 'rb')
      indexs = pickle.load(outfile)
      outfile.close()
      outfile = open('../Data/indexs_inv.pkl', 'rb')
      indexs_inv = pickle.load(outfile)
      outfile.close()
      outfile = open('../Data/labels.pkl', 'rb')
      labels = pickle.load(outfile)
      outfile.close()
      outfile = open('../Data/y_name.pkl', 'rb')
      y_name = pickle.load(outfile)
      outfile.close()
    else:
      # load datas
      labels, y_name = rf.read_label(label_path)
      dates, datas, indexs, indexs_inv = rf.read_data(data_path, y_name, 0)
      # save datas
      np.save('../Data/datas.npy', datas)
      np.save('../Data/dates.npy', dates)
      outfile = open('../Data/indexs.pkl', 'wb')
      pickle.dump(indexs, outfile)
      outfile.close()
      outfile = open('../Data/indexs_inv.pkl', 'wb')
      pickle.dump(indexs_inv, outfile)
      outfile.close()
      outfile = open('../Data/labels.pkl', 'wb')
      pickle.dump(labels, outfile)
      outfile.close()
      outfile = open('../Data/y_name.pkl', 'wb')
      pickle.dump(y_name, outfile)
      outfile.close()
    return dates, datas, indexs, indexs_inv

def write():
    target = open("small_var.txt",'w')
    target.truncate()
    dates, datas, indexs = rf.read_data(StatObj.data_path(), 0.5);
    fp_label, y_name = rf.read_label(StatObj.label_path())
    a = np.ones()
    for name in y_name:
        id_value= indexs[name]
        lines = str(datas[:,id_value])
        target.write(lines[1:-2])
        target.write('\n')
    target.close()

def main():
    dates, datas, indexs, indexs_inv = read_fast()
    print(type(datas))
    
# a = np.asarray()
# np.savetxt("foo.csv", a, delimiter=",")
main()
