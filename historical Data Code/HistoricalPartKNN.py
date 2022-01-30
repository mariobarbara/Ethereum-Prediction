import numpy as np
import pandas as pd
from math import *
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


if __name__ == '__main__':
    dataset = pd.read_csv("dataset.csv")
    l1 = dataset['Label1']
    l3 = dataset['Label3']
    l5 = dataset['Label5']

    #feat = ['open', 'Volume USDT', 'close_SMA_50', 'open_SMA_50', 'high_SMA_50', 'low_WMA_10', 'atr_ema_based','AverageGasPrice_avg']
    feat = ['Volume USDT', 'AverageBlockTime_avg', 'AverageDailyTransactionFee_avg',
            'AverageDailyTransactionFee_SMA_10', 'UniqueAddressesCount_avg', 'DailyBlockRewards_avg',
            'DailyTransactions_SMA_50','low_WMA_50', 'DailyTransactions_SMA_10','close_WMA_50',
            'AverageDailyTransactionFee_SMA_50','NetworkHashrate_SMA_10','close_WMA_20','NetworkHashrate_avg',
            'DailyVerifiedContracts _avg']

    for l in list(dataset.columns):
        if l not in feat:
            dataset.drop(l, axis=1, inplace=True)
    for n in range(3, 25, 2):
        knn_classifier = KNeighborsClassifier(n_neighbors=n)
        scaler = MinMaxScaler()
        curr, X_test, curr_y, y_test = train_test_split(dataset, l5, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, test_size=0.25, shuffle=False)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        knn_classifier.fit(X_train, y_train)
        prediction = knn_classifier.predict(X_val)
        result_matrix = confusion_matrix(y_val, prediction)
        accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
        print(accuracy)



