from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
from math import *
import matplotlib.pyplot as plt
from numpy import size
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


def calcACC(prediction, open_vals, actual_prediction) -> float:
    acc = 0
    for i in range(1, size(prediction)):
        if prediction[i] > open_vals[i-1] and actual_prediction[i-1] == 1:
            acc += 1
        elif prediction[i] < open_vals[i-1] and actual_prediction[i-1] == -1:
            acc += 1
        if 0.99*open_vals[i-1] <= prediction[i] <= 1.01*open_vals[i-1] and actual_prediction[i-1] == 0:
            acc += 1
    return acc/size(prediction)


if __name__ == '__main__':
    dataset = pd.read_csv("dataset.csv")
    l1 = dataset['Label1']
    l3 = dataset['Label3']
    l5 = dataset['Label5']
    data_open = dataset['open']
    data_date = dataset['Date(UTC)']
    # feat = ['open', 'Volume USDT', 'close_SMA_50', 'open_SMA_50', 'high_SMA_50', 'low_WMA_10', 'atr_ema_based','AverageGasPrice_avg']
    feat = ['Volume USDT', 'AverageBlockTime_avg', 'AverageDailyTransactionFee_avg',
            'AverageDailyTransactionFee_SMA_10', 'UniqueAddressesCount_avg', 'DailyBlockRewards_avg',
            'DailyTransactions_SMA_50', 'low_WMA_50', 'DailyTransactions_SMA_10', 'close_WMA_50',
            'AverageDailyTransactionFee_SMA_50', 'NetworkHashrate_SMA_10', 'close_WMA_20', 'NetworkHashrate_avg',
            'DailyVerifiedContracts _avg']
    days = list()

    d0 = date(2017, 10, 16)
    for day in data_date:
        d1 = datetime.strptime(day, '%m/%d/%Y')
        d1 = d1.date()
        delta = d1 - d0
        days.append(delta.days)

    for l in list(dataset.columns):
        if l not in feat:
            dataset.drop(l, axis=1, inplace=True)

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    #scaler = RobustScaler()
    curr, X_test, curr_y, y_test = train_test_split(dataset, data_open, train_size=0.8, test_size=0.2, shuffle=False)

    # ********************************************
    # the following split is for plotting purposes
    # ********************************************
    curr_day, day_test, curr_open, open_test = train_test_split(days, data_open, train_size=0.8, test_size=0.2, shuffle=False)

    # ***********************************************
    # the following split is for calculating accuracy
    # ***********************************************
    curr_data, data_test, curr_label, label_test = train_test_split(dataset, l1, train_size=0.8, test_size=0.2, shuffle=False)

    X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, train_size=0.75, test_size=0.25, shuffle=False)
    day_train, day_val, open_train, open_val = train_test_split(curr_day, curr_open, train_size=0.75, test_size=0.25, shuffle=False)
    data_train, data_val, label_train, label_val = train_test_split(curr_data, curr_label, train_size=0.75, test_size=0.25, shuffle=False)

    X_train = scaler.fit_transform(X_train)
    curr = scaler.fit_transform(curr)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    c_values = [0.001, 0.01, 1.0, 1000.0, 10000.0]
    gamma_vals = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    best_lin_acc = 0
    best_lin_gamma = 0
    best_c_lin = 0
    # create and train a linear svr model
    for c in c_values:
        for gamma in gamma_vals:
            lin_svr = SVR(kernel='linear', C=c, gamma=gamma)
            lin_svr.fit(X_train, y_train)
            lin_predict_list = lin_svr.predict(X_val)
            lin_acc = calcACC(lin_predict_list, list(open_val), list(label_val))
            if lin_acc > best_lin_acc:
                best_lin_gamma = gamma
                best_lin_acc = lin_acc
                best_c_lin = c
    best_lin_svr = SVR(kernel='linear', C=best_c_lin, gamma=best_lin_gamma)
    best_lin_svr.fit(curr, curr_y)
    print("lin svr values: (C-Value, gamma):", best_c_lin, best_lin_gamma)

    # ***************************************
    # create and train a polynomial svr model
    # ***************************************
    # degree_vals = [1, 2, 3, 4, 5]
    # best_poly_acc = 0
    # best_c_poly = 0
    # best_degree_poly = 0
    # best_gamma_poly = 0
    # for c in c_values:
    #     for degree in degree_vals:
    #         for gamma in gamma_vals:
    #             poly_svr = SVR(kernel='poly', C=c, degree=degree, gamma=gamma)
    #             poly_svr.fit(X_train, y_train)
    #             poly_predict_list = poly_svr.predict(X_val)
    #             poly_acc = calcACC(poly_predict_list, list(open_val), list(label_val))
    #             if poly_acc > best_poly_acc:
    #                 best_poly_acc = poly_acc
    #                 best_c_poly = c
    #                 best_degree_poly = degree
    #                 best_gamma_poly = gamma
    # best_poly_svr = SVR(kernel='poly', C=best_c_poly, degree=best_degree_poly, gamma=best_gamma_poly)
    # best_poly_svr.fit(curr, curr_y)
    # print("poly svr values: (degree, C-Value, gamma):", best_degree_poly, best_c_poly, best_gamma_poly)

    # ********************************
    # create and train a RBF svr model
    # ********************************
    # best_rbf_acc = 0
    # best_c_rbf = 0
    # best_gamma_rbf = 0
    # for c in c_values:
    #     for gamma in gamma_vals:
    #         rbf_svr = SVR(kernel='rbf', C=c, gamma=gamma)
    #         rbf_svr.fit(X_train, y_train)
    #         rbf_predict_list = rbf_svr.predict(X_val)
    #         rbf_acc = calcACC(rbf_predict_list, list(open_val), list(label_val))
    #         if rbf_acc > best_rbf_acc:
    #             best_rbf_acc = rbf_acc
    #             best_c_rbf = c
    #             best_gamma_rbf = gamma
    # best_rbf_svr = SVR(kernel='rbf', C=best_c_rbf, gamma=best_gamma_rbf)
    # best_rbf_svr.fit(curr, curr_y)
    # print("rbf svr values: (gamma, C-Value):", best_gamma_rbf, best_c_rbf)

    # ********************************
    # create and train a sigmoid svr model
    # ********************************
    # best_sigmoid_acc = 0
    # best_c_sigmoid = 0
    # best_gamma_sigmoid = 0
    # for c in c_values:
    #     for gamma in gamma_vals:
    #         sigmoid_svr = SVR(kernel='sigmoid', C=c, gamma=gamma)
    #         sigmoid_svr.fit(X_train, y_train)
    #         sigmoid_predict_list = sigmoid_svr.predict(X_val)
    #         sigmoid_acc = calcACC(sigmoid_predict_list, list(open_val), list(label_val))
    #         if sigmoid_acc > best_sigmoid_acc:
    #             best_sigmoid_acc = sigmoid_acc
    #             best_c_sigmoid = c
    #             best_gamma_sigmoid = gamma
    # best_sigmoid_svr = SVR(kernel='sigmoid', C=best_c_sigmoid, gamma=best_gamma_sigmoid)
    # best_sigmoid_svr.fit(curr, curr_y)
    # print("sigmoid svr values: (gamma, C-Value):", best_gamma_sigmoid, best_c_sigmoid)



    # *************************************************
    # plot the models to check the fits the data better
    # *************************************************
    plt.figure(figsize=(16, 8))
    plt.scatter(day_train, open_train, color='red', label='data', s=3)
    # plt.plot(day_train, best_rbf_svr.predict(X_train), color='green', label='RBF Model')
    # plt.plot(day_train, best_poly_svr.predict(X_train), color='orange', label='poly Model')
    plt.plot(day_train, best_lin_svr.predict(X_train), color='blue', label='linear Model')
    # plt.plot(day_train, best_sigmoid_svr.predict(X_train), color='black', label='sigmoid Model')
    plt.xlabel("day (started count from 16.10.2017)")
    plt.ylabel("Close price")
    plt.legend()
    plt.show()

    # rbf_predict_list = best_rbf_svr.predict(X_test)
    # poly_predict_list = best_poly_svr.predict(X_test)
    lin_predict_list = best_lin_svr.predict(X_test)
    # sigmoid_predict_list = best_sigmoid_svr.predict(X_test)

    # rbf_acc = calcACC(rbf_predict_list, list(y_test), list(label_test))
    # poly_acc = calcACC(poly_predict_list, list(y_test), list(label_test))
    lin_acc = calcACC(lin_predict_list, list(y_test), list(label_test))
    # sigmoid_acc = calcACC(sigmoid_predict_list, list(y_test), list(label_test))

    # print('the rbf accuracy:', rbf_acc)
    # print('the poly accuracy:', poly_acc)
    print('the linear accuracy:', lin_acc)
    # print('the sigmoid accuracy:', sigmoid_acc)
