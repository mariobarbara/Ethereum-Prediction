import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import TimeSeriesSplit





"""

@author: Nawras
"""

"""
SFS: Sequential Forward Selection
starts with empty set of features and returns the K-best feature according
to the highest scores the  clf param classifier gave.

    @param clf: scikit-learn classifier or regressor
    @param k_features: Number of features to select, where k_features < the full feature set
    @param X_train: training samples (dataframe)
    @param y_train: training labels (dataframe)
    @param X_val: validation samples (dataframe)
    @param y_val: validation labels (dataframe)
    @param all_features: list of all the features in the training and validation

    @return selected_features_subset: list of size=k_features, for the top-k scored features subset
    @return top_subset: list of the best scored features subset in case  top_subset.size < selected_features_subset.size
    """
from sklearn.metrics import accuracy_score


def score_calc(clf, X_train, X_val, y_train, y_val):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    # print("Accuracy RandomForestClassifier:")
    score = accuracy_score(y_val, y_pred)
    return score


def sfs(clf, k_features, X_train, y_train, X_val, y_val, all_features):
    selected_features_subset = []
    dim = 0
    max_score = -1
    top_subset = []
    top_dim = 0
    while dim < k_features:
        dim_max_score = -1
        for feature in all_features:
            if feature not in selected_features_subset:
                current_subset = selected_features_subset.copy()
                current_subset.append(feature)
                # print(current_subset)
                cur_score = score_calc(clf, X_train[:, current_subset], X_val[:, current_subset], y_train, y_val)
                if cur_score > dim_max_score:
                    dim_max_score = cur_score
                    top_feature = feature
                    # print(top_feature)
        # check if there is new feature to add:
        selected_features_subset.append(top_feature)
        dim += 1
        if (dim_max_score >= max_score):
            max_score = dim_max_score
            top_subset = selected_features_subset
            top_dim = dim
        # else:
        # print("dim=", dim, " current score is:", max_score, "higher score with subset of dim:", top_dim,
        #       "with score=", max_score)
    # print(max_score)
    return selected_features_subset, top_subset, max_score


def printFeatures(features_list, feature_indices):
    features_to_print = '['
    for index in feature_indices:
        features_to_print = features_to_print + features_list[index] + ', '
    features_to_print = features_to_print + ']'
    print(features_to_print)




if __name__ == '__main__':
    dataset = pd.read_csv("dataset_tweets_part.csv")
    l1 = dataset['label1']
    l = dataset['label']
    dataset.drop('label', axis=1, inplace=True)
    dataset.drop('label1', axis=1, inplace=True)

    # # Features to remove according to Correlation Experiment
    # # features_to_remove = ["Likes[0,1000]Compound[0.1, 1)", "Followers[0,1000]Compound[0, 0.1)",
    # #                       "Likes[0,1000]Compound[-1, -0.1)"]
    #
    # # for feature in features_to_remove:
    # #     dataset.drop(l, axis=1, inplace=True)
    #


    ##################################################################################################
    # Confusion matrix for selected features for (-1,0,1) classification
    ##################################################################################################
    all_features = list(dataset.columns)
    features = [45, 57, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 61, 2, 49, 0, 41, 3, 32, 42]
    features_names = [all_features[i] for i in features]
    for feature in all_features:
        if feature not in features_names:
            dataset.drop(feature, axis=1, inplace=True)
    scaler = MinMaxScaler()
    curr, X_test, curr_y, y_test = train_test_split(dataset, l, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, test_size=0.25, shuffle=False)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    knn_classifier = KNeighborsClassifier(n_neighbors=18)
    knn_classifier.fit(X_train, y_train)
    plot_confusion_matrix(knn_classifier, X_val, y_val)
    plt.title('Confusion Matrix - Validation Set - Classifications (-1,1)')
    plt.show()

    ##################################################################################################
    # Confusion matrix for selected features for (-1,1) classification
    ##################################################################################################
    # all_features = list(dataset.columns)
    # features = [20, 43, 46, 57, 47, 48, 51, 54, 55, 56, 58, 59, 60, 61, 50]
    # features_names = [all_features[i] for i in features]
    # for feature in all_features:
    #     if feature not in features_names:
    #         dataset.drop(feature, axis=1, inplace=True)
    # scaler = MinMaxScaler()
    # curr, X_test, curr_y, y_test = train_test_split(dataset, l1, test_size=0.2, shuffle=False)
    # X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, test_size=0.25, shuffle=False)
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    # knn_classifier = KNeighborsClassifier(n_neighbors=23)
    # knn_classifier.fit(X_train, y_train)
    # plot_confusion_matrix(knn_classifier, X_val, y_val)
    # plt.title('Confusion Matrix - Validation Set - Classifications (-1,0,1)')
    # plt.show()

    ##################################################################################################
    # Test for (-1,1) classification
    ##################################################################################################
    # all_features = list(dataset.columns)
    # features = [45, 57, 47, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 61, 2, 49, 0, 41, 3, 32, 42]
    # features_names = [all_features[i] for i in features]
    # for feature in all_features:
    #     if feature not in features_names:
    #         dataset.drop(feature, axis=1, inplace=True)
    # scaler = MinMaxScaler()
    # curr, X_test, curr_y, y_test = train_test_split(dataset, l, test_size=0.2, shuffle=False)
    # curr = scaler.fit_transform(curr)
    # X_test = scaler.transform(X_test)
    # knn_classifier = KNeighborsClassifier(n_neighbors=18)
    # knn_classifier.fit(curr, curr_y)
    # plot_confusion_matrix(knn_classifier, X_test, y_test)
    # prediction = knn_classifier.predict(X_test)
    # result_matrix = confusion_matrix(y_test, prediction)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1]) / np.sum(result_matrix, initial=0)
    # print(accuracy)
    # plt.title('Confusion Matrix - Test Set - Classifications (-1,1)')
    # plt.show()

    #################################################################################################
    # Confusion matrix for selected features for (-1,0,1) classification
    #################################################################################################
    all_features = list(dataset.columns)
    features = [20, 43, 46, 57, 47, 48, 51, 54, 55, 56, 58, 59, 60, 61, 50]
    features_names = [all_features[i] for i in features]
    for feature in all_features:
        if feature not in features_names:
            dataset.drop(feature, axis=1, inplace=True)
    scaler = MinMaxScaler()
    curr, X_test, curr_y, y_test = train_test_split(dataset, l1, test_size=0.2, shuffle=False)
    curr = scaler.fit_transform(curr)
    X_test = scaler.transform(X_test)
    knn_classifier = KNeighborsClassifier(n_neighbors=23)
    knn_classifier.fit(curr, curr_y)
    plot_confusion_matrix(knn_classifier, X_test, y_test)
    prediction = knn_classifier.predict(X_test)
    result_matrix = confusion_matrix(y_test, prediction)
    accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    print(accuracy)
    plt.title('Confusion Matrix - Test Set - Classifications (-1,0,1)')
    plt.show()


    # scaler = MinMaxScaler()
    # curr, X_test, curr_y, y_test = train_test_split(dataset, l, test_size=0.1, shuffle=False)
    # X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, test_size=0.25, shuffle=False)
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    # knn_classifier = KNeighborsClassifier(n_neighbors=18)
    # knn_classifier.fit(X_train, y_train)
    #
    # accuracies = []
    # model = []
    # max_accuracy = 0
    # max_model = ''
    # for n in range(15, 50):
    #     print('num of neighbors is ' + str(n))
    #     knn_classifier = KNeighborsClassifier(n_neighbors=n)
    #     knn_classifier.fit(X_train, y_train)
    #     for num in range(15, 25):
    #         x, y, score = sfs(knn_classifier, num, X_train, y_train, X_val, y_val,
    #                    [i for i in range(0, len(list(dataset.columns)))])
    #         # printFeatures(list(dataset.columns), x)
    #         print(x)
    #         model.append('(' + str(n) + ', ' + str(num) + ')')
    #         accuracies.append(score)
    #         if score > max_accuracy:
    #             max_accuracy = score
    #             max_model = '(' + str(n) + ', ' + str(num) + ')'
    #         # print(y)
    #         # print(result_matrix)
    #         # print(score)
    # print('Best accuracy')
    # print(max_accuracy)
    # print('Best Model')
    # print(max_model)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(model, accuracies)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()
    # # for i in range(0, 400):
    # #     model.append('(' + str(i) + ',' + str(i) + ')')
    # #     accuracies.append(0.4)
    # # # plt.plot(model, accuracies)
    # # fig, ax = plt.subplots(1, 1)
    # # ax.plot(model, accuracies)
    # # ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    # # plt.xticks(rotation=90)
    # # plt.tight_layout()
    # # plt.show()
    # # plt.coord_flip()
    # # plt.xticks(rotation=90)
    # # plt.tight_layout()
    # # plt.show()
    #
    # # knn_classifier = KNeighborsClassifier(n_neighbors=18)
    # # knn_classifier.fit(X_train, y_train)
    # # x, y = sfs(knn_classifier, 21, X_train, y_train, X_val, y_val, [i for i in range(0, len(list(dataset.columns)))])
    # # X_train, X_val, X_test = X_train[:, x], X_val[:, x], X_test[:,x]
    # # curr = scaler.fit_transform(curr)
    # # curr = curr[:, x]
    # # knn_classifier.fit(curr, curr_y)
    # # predict = knn_classifier.predict(X_test)
    # # score = accuracy_score(y_test, predict)
    # # result_matrix = confusion_matrix(y_test, predict)


#####################################################################################3
    # accuracies = []
    # model = []
    # max_accuracy = 0
    # max_model = ''
    # for n in range(15, 50):
    #     # print('num of neighbors is ' + str(n))
    #     knn_classifier = KNeighborsClassifier(n_neighbors=n)
    #     for num in range(15, 25):
    #         scaler = MinMaxScaler()
    #         curr, X_test, curr_y, y_test = train_test_split(dataset, l, test_size=0.2, shuffle=False)
    #         X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, test_size=0.25, shuffle=False)
    #         X_train = scaler.fit_transform(X_train)
    #         X_val = scaler.transform(X_val)
    #         X_test = scaler.transform(X_test)
    #         knn_classifier.fit(X_train, y_train)
    #         x, y, score = sfs(knn_classifier, num, X_train, y_train, X_val, y_val,
    #                           [i for i in range(0, len(list(dataset.columns)))])
    #         print(x)
    #         curr_accuracy = []
    #         model.append('(' + str(n) + ', ' + str(num) + ')')
    #         print('(' + str(n) + ', ' + str(num) + ')')
    #         curr_dataset = dataset.iloc[:, x]
    #         tscv = TimeSeriesSplit()
    #         TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=0.2)
    #         for train_index, test_index in tscv.split(curr_dataset):
    #             curr, X_test = curr_dataset.iloc[train_index], curr_dataset.iloc[test_index]
    #             y_curr, y_test = l.iloc[train_index], l.iloc[test_index]
    #             X_train, X_val, y_train, y_val = train_test_split(curr, y_curr, test_size=0.25, shuffle=False)
    #             scaler = MinMaxScaler()
    #             X_train = scaler.fit_transform(X_train)
    #             X_val = scaler.transform(X_val)
    #             X_test = scaler.transform(X_test)
    #             knn_classifier = KNeighborsClassifier(n_neighbors=n)
    #             knn_classifier.fit(X_train, y_train)
    #             prediction = knn_classifier.predict(X_val)
    #             result_matrix = confusion_matrix(y_val, prediction)
    #             score = (result_matrix[0][0] + result_matrix[1][1]) / np.sum(result_matrix, initial=0)
    #             curr_accuracy.append(score)
    #             print(score)
    #         acc = sum(curr_accuracy)/len(curr_accuracy)
    #         print(acc)
    #         accuracies.append(acc)
    #         if acc > max_accuracy:
    #             max_accuracy = acc
    #             max_model = '(' + str(n) + ', ' + str(num) + ')'
    #             # print(y)
    #             # print(result_matrix)
    #             # print(score)
    # print('Best accuracy')
    # print(max_accuracy)
    # print('Best Model')
    # print(max_model)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(model, accuracies)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()


