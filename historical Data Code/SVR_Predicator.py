import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR
import pandas as pd





if __name__ == "__main__":

    training_data_set = pd.read_csv("1Hour_LastMonth.csv",dtype={"date":"string","close":float,"open":float,"label":int})
    to_pred = training_data_set.head(2)
    training_data_set = training_data_set.iloc[::-1]

    hours = list()
    close_data = list()
    j = 0
    for i in range(len(training_data_set)-1,1 ,-1):
        hours.append([j])
        j += 1
        close_data.append(float(training_data_set.at[i, 'close']))


    #lin_svr = SVR(kernel='linear',C=1000.0)
    rbf_svr = SVR(kernel='rbf', C=1000, gamma=0.3)
    rbf_svr.fit(hours, close_data)
    close = rbf_svr.predict([[len(training_data_set)-2],[len(training_data_set)-1]])
    print(close)
    print(to_pred['close'])