import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from scipy.stats.stats import pearsonr
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from matplotlib.pyplot import figure
# from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif



if __name__ == '__main__':
    all_data = pd.read_csv('dataset_after_variance.csv')
    labels = pd.read_csv("dataset.csv")
    l1 = labels['Label1']
    l3 = labels['Label3']
    l5 = labels['Label5']
    all_data.drop("Unnamed: 0", axis=1, inplace=True)

    # knn_classifier = KNeighborsClassifier(n_neighbors=11)
    # scaler = MinMaxScaler()
    # curr, X_test, curr_y, y_test = train_test_split(all_data, l3, test_size=0.2, shuffle=False)
    # X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, test_size=0.25, shuffle=False)
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # knn_classifier.fit(X_train, y_train)
    # k_s = []
    # scores = []
    # max_accuracy = 0
    # best_features = []
    # for k in range(1, 30):
    #     sfs = SFS(knn_classifier, k_features=k, forward=True,
    #               floating=False,
    #               scoring='accuracy',
    #               cv=None)
    #     sfs.fit(X_train, y_train, custom_feature_names=list(all_data.columns))
    #     if sfs.k_score_ > max_accuracy:
    #         max_accuracy = sfs.k_score_
    #         best_features = sfs.k_feature_names_
    #     print(sfs.k_score_)
    #     print(sfs.k_feature_names_)
    #     k_s.append(k)
    #     scores.append(sfs.k_score_)
    # plt.plot(k_s, scores, 'o')
    # plt.show()
    # print(skb.k_score_)
    # print(skb.k_feature_names_)
    # print(scores)

    # Create a SelectKBest object
    # chi2 or MI


    scaler = MinMaxScaler()
    curr, X_test, curr_y, y_test = train_test_split(all_data, l3, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, test_size=0.25, shuffle=False)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    MI_score = mutual_info_classif(X_train, y_train, random_state=0)

    skb = SelectKBest(score_func=mutual_info_classif, k='all')  # Set f_classif as our criteria to select features

    fit = skb.fit(X_train, y_train)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(all_data.columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    best_features = featureScores.nlargest(20, 'Score')
    print(best_features['Specs'])  # print 5 best features






