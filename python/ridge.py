from sklearn.linear_model import Ridge
import read_file as rf
import numpy as np

data_path = "../Data_M.csv"
label_path = "../Dico_M.csv"

labels, y_name = rf.read_label(label_path)
dates, datas, indexs = rf.read_data(data_path, 0.3)

# seperate feature(x_train) and prediction(y_train)
y_index = [indexs[i] for i in y_name]
y_train = datas[:, y_index]
x_train = np.delete(datas, y_index, axis = 1)



clf = Ridge(alpha=1.0)
clf.fit(x_train[:-2, :], y_train[2:, 0])
print(clf.score(x_train[:-2, :], y_train[2:, 0]))
print(clf.coef_)
