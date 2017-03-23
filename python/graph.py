from static import *
import read_file as rf
import matplotlib.pyplot as plt
#
# labels, y_name = rf.read_label(StatObj.label_path())
# print('-------------------------------')
# print(type(labels))


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
    dates, datas, indexs = rf.read_data(StatObj.data_path(), 0.5);
    fp_label, y_name = rf.read_label(StatObj.label_path())
    line = [0]*len(y_name)
    for i,name in enumerate(y_name):
        id_value= indexs[name]
        line = plt.plot(datas[:,id_value], label = str(id_value))
        plt.legend(str(id_value))
    plt.show()

a = np.asarray()
np.savetxt("foo.csv", a, delimiter=",")
main()
