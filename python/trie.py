from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
import read_file as rf
import numpy as np
import argparse
import os, sys
import pickle
from static import *
import read_file as rf
import matplotlib.pyplot as plt

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

def write():
      labels, y_name = read_label(label_path)
      dates, datas, indexs, indexs_inv = read_data(data_path, 0.3)

def main():
    dates, datas, indexs, indexs_inv = rf.read_data(StatObj.data_path(), 0.5);
    fp_label, y_name = rf.read_label(StatObj.label_path())
    line = [0]*len(y_name)
    for i,name in enumerate(y_name):
        id_value= indexs[name]
        line = plt.plot(datas[:,id_value], label = str(id_value))
        plt.legend(str(id_value))
    plt.show()

# a = np.asarray()
# np.savetxt("foo.csv", a, delimiter=",")
main()
