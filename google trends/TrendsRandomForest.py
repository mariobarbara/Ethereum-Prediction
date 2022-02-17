import copy

import matplotlib.pyplot as plt
import tweepy as tw
import pandas as pd
import numpy as np
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.regressor import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

# def hyper_parameter_tuning(values, ):
#     accuracies = []
#     max_accuracy = 0
#     max_param = values[0]
#     for val in values:
#         clf = RandomForestClassifier(criterion=val, random_state=0)
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_val)
#         result_matrix = confusion_matrix(y_val, y_pred)
#         accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
#         if accuracy > max_accuracy:
#             max_accuracy = accuracy
#             max_param = cri
#         accuracies.append(accuracy)
#     plt.plot(criterions, accuracies)
#     plt.title('Tuning - Criterions')
#     plt.show()
#     clf = RandomForestClassifier(criterion=max_param, random_state=0)
#     clf.fit(X_train, y_train)
#     plot_confusion_matrix(clf, X_val, y_val)
#     plt.title('Confusion Matrix Valiation Set - num of criteion')
#     plt.show()
#     print(max_param)
#     print(max_accuracy)


if __name__ == '__main__':
    all_data = pd.read_csv("google_trends_dataset_processed.csv")
    l1 = all_data['label1']
    for l in ['label1']:
        all_data.drop(l, axis=1, inplace=True)

    features = list(all_data.columns)
    # top_20_features = ['ethereum_unscaled_prev', 'ethereum_prev', 'log_value_prev', 'ethereum_WMA_10', 'var_last_14',
    #                    'ethereum_SMA_20', 'ethereum_SMA_10', 'var_last_30', 'ethereum_SMA_14', 'Day', 'IQR_last_30',
    #                    'ethereum_SMA_50', 'Weekday', 'median_last_14', 'Q3_last_30', 'Q1_last_30',  'Q1_last_50',
    #                    'range_last_14', 'median_last_30', 'Q3_last_50']
    #
    # #Dropping all features that are out of TOP 20 imporatant features
    # for l in features:
    #     if l not in top_20_features:
    #         all_data.drop(l, axis=1, inplace=True)

    # X_train, X_test, y_train, y_test = train_test_split(all_data, l1, test_size=0.2, shuffle=False)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    #
    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)

    #Getting best 20 features according to features importance

    # clf = RandomForestClassifier(n_estimators=100)
    # #Train the model using the training sets y_pred=clf.predict(X_test)
    # clf.fit(X_train, y_train)
    # feature_imp = pd.Series(clf.feature_importances_, index=list(all_data.columns)).sort_values(ascending=False)
    # # Creating a bar plot
    # sns.set(rc={'figure.figsize': (15, 15)})
    # sns.barplot(x=feature_imp, y=feature_imp.index)
    # # Add labels to your graph
    # plt.xlabel('Feature Importance Score')
    # plt.ylabel('Features')
    # plt.title("Visualizing Important Features")
    # plt.legend()
    # plt.show()

    # Ranking the 20 features
    # features = list(all_data.columns)
    # m = RFECV(RandomForestClassifier(), scoring='accuracy')
    # m.fit(X_train, y_train)
    # features_ranking = m.ranking_
    # top_features = ['','','','','','','','','','','','','','','','','','','','']
    # for i in range(0,20):
    #     curr = features_ranking[i] - 1
    #     top_features[curr] = features[i]
    # print(top_features)


    # Checking best number of features
    # number_of_features = []
    # accuracies = []
    # for i in range(1, 20):
    #     number_of_features.append(str(i))
    #     curr_dataset = copy.deepcopy(all_data)
    #     for j in range(i, 20):
    #         curr_dataset.drop(top_features[j], axis=1, inplace=True)
    #     X_train, X_test, y_train, y_test = train_test_split(curr_dataset, l1, test_size=0.2, shuffle=False)
    #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)
    #
    #     scaler = StandardScaler().fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_val = scaler.transform(X_val)
    #     X_test = scaler.transform(X_test)
    #
    #     clf = RandomForestClassifier(n_estimators=100)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     accuracies.append(accuracy)
    # plt.plot(number_of_features, accuracies)
    # plt.show()


    last_features_to_use = ['ethereum_WMA_10', 'ethereum_SMA_20', 'var_last_14', 'ethereum_prev', 'var_last_30',
                            'ethereum_unscaled_prev']
    for l in features:
        if l not in last_features_to_use:
            all_data.drop(l, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(all_data, l1, test_size=0.1, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    clf5 = RandomForestClassifier(max_depth=4, n_estimators=36, random_state=0)
    clf5.fit(X_train, y_train)

    y_pred = clf5.predict(X_test)
    result_matrix = confusion_matrix(y_test, y_pred)
    f = sns.heatmap(result_matrix, annot=True, cmap="viridis")
    accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    print(accuracy)
    # plot_confusion_matrix(y_test, y_pred)
    plt.title('Confusion Matrix Test Set - Google Trends')
    plt.show()


    # number_of_trees = [i for i in range(1,500)]
    # accuracies = []
    # max_accuracy = 0
    # max_param = 10
    # best_clf = RandomForestClassifier()
    # for num in number_of_trees:
    #     clf = RandomForestClassifier(n_estimators=num, random_state=0)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         max_param = num
    #         best_clf = clf
    #     accuracies.append(accuracy)
    # # clf = RandomForestClassifier(n_estimators=max_param, random_state=0)
    # # clf.fit(X_train, y_train)
    # plt.plot(number_of_trees, accuracies)
    # plt.title('Tuning - Number of Trees')
    # plt.show()
    # plot_confusion_matrix(best_clf, X_val, y_val)
    # plt.title('Confusion Matrix Valiation Set - num of estimators')
    # plt.show()
    # print(max_param)
    # print(max_accuracy)
    #
    # criterions = ['gini', 'entropy']
    # accuracies = []
    # max_accuracy = 0
    # max_param = 'gini'
    # for cri in criterions:
    #     clf = RandomForestClassifier(criterion=cri, random_state=0)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         max_param = cri
    #     accuracies.append(accuracy)
    # plt.plot(criterions, accuracies)
    # plt.title('Tuning - Criterions')
    # plt.show()
    # clf = RandomForestClassifier(criterion=max_param, random_state=0)
    # clf.fit(X_train, y_train)
    # plot_confusion_matrix(clf, X_val, y_val)
    # plt.title('Confusion Matrix Valiation Set - num of criteion')
    # plt.show()
    # print(max_param)
    # print(max_accuracy)
    #
    # max_depths = [i for i in range(2, 30)]
    # accuracies = []
    # max_accuracy = 0
    # max_param = 2
    # best_clf = RandomForestClassifier()
    # for num in max_depths:
    #     clf = RandomForestClassifier(max_depth=num, random_state=0)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         max_param = num
    #         best_clf = clf
    #     accuracies.append(accuracy)
    # plt.plot(max_depths, accuracies)
    # plt.title('Tuning - Max Depths')
    # plt.show()
    # # clf = RandomForestClassifier(max_depth=max_param, random_state=0)
    # # clf.fit(X_train, y_train)
    # plot_confusion_matrix(best_clf, X_val, y_val)
    # plt.title('Confusion Matrix Valiation Set - Max Depths')
    # plt.show()
    # print(max_param)
    # print(max_accuracy)
    #
    # max_leaf_nodes = [i for i in range(2, 500)]
    # accuracies = []
    # max_accuracy = 0
    # max_param = 2
    # best_clf = RandomForestClassifier()
    # for num in max_leaf_nodes:
    #     clf = RandomForestClassifier(max_leaf_nodes=num, random_state=0)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         max_param = num
    #         best_clf = clf
    #     accuracies.append(accuracy)
    # plt.plot(max_leaf_nodes, accuracies)
    # plt.title('Tuning - Max Leaf Nodes')
    # plt.show()
    # # clf = RandomForestClassifier(max_leaf_nodes=max_param, random_state=0)
    # # clf.fit(X_train, y_train)
    # plot_confusion_matrix(best_clf, X_val, y_val)
    # plt.title('Confusion Matrix Valiation Set - Max Leaf Nodes')
    # plt.show()
    # print(max_param)
    # print(max_accuracy)

    # min_samples_split = [i for i in range(2, 500)]
    # accuracies = []
    # max_accuracy = 0
    # max_param = 2
    # best_clf = RandomForestClassifier()
    # for num in min_samples_split:
    #     clf = RandomForestClassifier(min_samples_split=num, random_state=0)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         max_param = num
    #         best_clf = clf
    #     accuracies.append(accuracy)
    # plt.plot(min_samples_split, accuracies)
    # plt.title('Tuning - Min Sample Split')
    # plt.show()
    # # clf = RandomForestClassifier(max_leaf_nodes=max_param, random_state=0)
    # # clf.fit(X_train, y_train)
    # plot_confusion_matrix(best_clf, X_val, y_val)
    # plt.title('Confusion Matrix Valiation Set - Min Samples split')
    # plt.show()
    # print(max_param)
    # print(max_accuracy)

    # min_weight_fraction_leaf = [(i/100) for i in range(1, 50)]
    # accuracies = []
    # max_accuracy = 0
    # max_param = 2
    # best_clf = RandomForestClassifier()
    # for num in min_weight_fraction_leaf:
    #     clf = RandomForestClassifier(min_weight_fraction_leaf=num, random_state=0)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         max_param = num
    #         best_clf = clf
    #     accuracies.append(accuracy)
    # plt.plot(min_weight_fraction_leaf, accuracies)
    # plt.title('Tuning - min_weight_fraction_leaf')
    # plt.show()
    # # clf = RandomForestClassifier(max_leaf_nodes=max_param, random_state=0)
    # # clf.fit(X_train, y_train)
    # plot_confusion_matrix(best_clf, X_val, y_val)
    # plt.title('Confusion Matrix Valiation Set - min_weight_fraction_leaf')
    # plt.show()
    # print(max_param)
    # print(max_accuracy)


    # accuracies = []
    # max_accuracy = 0
    # max_param = 2
    # ccp_alphas = [1]
    # best_clf = RandomForestClassifier()
    # for num in ccp_alphas:
    #     clf = RandomForestClassifier(class_weight={-1:0, 0:1, 1:0}, random_state=0)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         max_param = num
    #         best_clf = clf
    #     accuracies.append(accuracy)
    # plt.plot(ccp_alphas, accuracies)
    # plt.title('Tuning - min_weight_fraction_leaf')
    # plt.show()
    # # clf = RandomForestClassifier(max_leaf_nodes=max_param, random_state=0)
    # # clf.fit(X_train, y_train)
    # plot_confusion_matrix(best_clf, X_val, y_val)
    # plt.title('Confusion Matrix Valiation Set - min_weight_fraction_leaf')
    # plt.show()
    # print(max_param)
    # print(max_accuracy)

    # clf0 = RandomForestClassifier(random_state=0)
    # clf1 = RandomForestClassifier(max_depth=4, random_state=0)
    # clf2 = RandomForestClassifier(min_weight_fraction_leaf=0.03, random_state=0)
    # clf3 = RandomForestClassifier(n_estimators=36, random_state=0)
    # clf4 = RandomForestClassifier(max_depth=4, min_weight_fraction_leaf=0.03, random_state=0)
    # clf5 = RandomForestClassifier(max_depth=4, n_estimators=36, random_state=0)
    # clf6 = RandomForestClassifier(min_weight_fraction_leaf=0.03, n_estimators=36, random_state=0)
    # clf7 = RandomForestClassifier(min_weight_fraction_leaf=0.03, n_estimators=36, max_depth=4, random_state=0)
    # classifiers = [clf0, clf1, clf2, clf3, clf4, clf5, clf6, clf7]
    # estimators_str = 'n:36,'
    # max_depth_str = 'depth:4,'
    # min_weight_fraction_leaf_str = 'fraction:0.03,'
    # random_state_str = 'random_state : 0}'
    # combos = ['{}', max_depth_str,
    #           min_weight_fraction_leaf_str,
    #           estimators_str,
    #           max_depth_str + min_weight_fraction_leaf_str,
    #           estimators_str + max_depth_str,
    #           estimators_str + min_weight_fraction_leaf_str,
    #           estimators_str + min_weight_fraction_leaf_str + max_depth_str]
    # accuracies = []
    # max_accuracy = 0
    # max_param = 2
    # best_clf = RandomForestClassifier()
    # for clf in classifiers:
    #     # clf = RandomForestClassifier(params)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_val)
    #     result_matrix = confusion_matrix(y_val, y_pred)
    #     accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         max_param = clf
    #         best_clf = clf
    #     accuracies.append(accuracy)
    # plt.plot(combos, accuracies)
    # plt.title('Tuning - Best Classifier')
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()
    # plot_confusion_matrix(best_clf, X_val, y_val)
    # plt.title('Confusion Matrix Valiation Set - Best Classifier')
    # plt.show()
    # print(max_param)
    # print(max_accuracy)




    # number_of_trees = [60, 70, 80, 90, 100, 150, 200]
    number_of_trees = [10, 20, 30]
    criterions = ['gini', 'entropy']
    # max_depths = [5, 10, 15, 20, 30, 40]
    max_depths = [1, 2, 3, 4, 5]
    # max_leaf_nodes = [5, 15, 25, 35]
    max_leaf_nodes = [2, 3, 4, 5]
    min_samples_split = [2, 5, 7]
    min_samples_leaf = [1, 2, 3]
    max_features = ['auto', 'sqrt']
    bootstrap = [False, True]

    param_grid = {
        'n_estimators': number_of_trees,
        'criterion': criterions,
        'max_depth': max_depths,
        'max_leaf_nodes': max_leaf_nodes,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap
    }

    # models = ['GridSearchCV', 'RandomizedSearchCV']
    # accuracies = []
    # rf_Model = RandomForestClassifier()
    # rf_Grid = GridSearchCV(estimator=rf_Model, param_grid=param_grid, cv=3, verbose=2, n_jobs=4)
    # rf_Grid.fit(X_train, y_train)
    # print(rf_Grid.best_params_)
    # print(rf_Grid.best_score_)
    # accuracies.append(rf_Grid.best_score_)
    #
    # rf_Model = RandomForestClassifier()
    # rf_Grid = RandomizedSearchCV(estimator=rf_Model, param_distributions=param_grid, cv=3, verbose=2, n_jobs=4)
    # rf_Grid.fit(X_train, y_train)
    # print(rf_Grid.best_params_)
    # print(rf_Grid.best_score_)
    # accuracies.append(rf_Grid.best_score_)
    # plt.plot(models, accuracies)
    # plt.show()



    # models = ['First', 'Second']
    # accuracies = []
    # rf = RandomForestClassifier(bootstrap=False, criterion='gini', max_depth=15, max_features='auto', max_leaf_nodes=5, min_samples_leaf=2, min_samples_split=2, n_estimators=70)
    # rf.fit(X_train,y_train)
    # pred1 = rf.predict(X_test)
    # result_matrix = confusion_matrix(y_test, pred1)
    # print(result_matrix)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    # print(accuracy)
    # plot_confusion_matrix(rf, X_test, y_test)
    # plt.show()
    # accuracies.append(accuracy)
    #
    # rf = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=5, max_features='auto', max_leaf_nodes=25, min_samples_leaf=1, min_samples_split=5, n_estimators=90)
    # rf.fit(X_train,y_train)
    # pred2 = rf.predict(X_test)
    # result_matrix = confusion_matrix(y_test, pred2)
    # print(result_matrix)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    # print(accuracy)
    # plot_confusion_matrix(rf, X_test, y_test)
    # plt.show()
    # accuracies.append(accuracy)
    #
    # rf = RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=10, max_features='auto', max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=5, n_estimators=70)
    # rf.fit(X_train,y_train)
    # pred2 = rf.predict(X_test)
    # result_matrix = confusion_matrix(y_test, pred2)
    # print(result_matrix)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    # print(accuracy)
    # plot_confusion_matrix(rf, X_test, y_test)
    # plt.show()
    # accuracies.append(accuracy)
    #
    # rf = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=5, max_features='auto', max_leaf_nodes=35, min_samples_leaf=2, min_samples_split=2, n_estimators=80)
    # rf.fit(X_train,y_train)
    # pred2 = rf.predict(X_test)
    # result_matrix = confusion_matrix(y_test, pred2)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    # print(result_matrix)
    # print(accuracy)
    # plot_confusion_matrix(rf, X_test, y_test)
    # plt.show()
    # accuracies.append(accuracy)
    #
    #

    # rf = RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=15, max_features='auto', max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=5, n_estimators=90)
    # rf.fit(X_train,y_train)
    # pred2 = rf.predict(X_test)
    # result_matrix = confusion_matrix(y_test, pred2)
    # print(result_matrix)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    # print(accuracy)
    # plot_confusion_matrix(rf, X_test, y_test)
    # plt.show()
    #



    # rf = RandomForestClassifier(n_estimators=500)
    # rf.fit(X_train,y_train)
    # pred1 = rf.predict(X_test)
    # result_matrix = confusion_matrix(y_test, pred1)
    # print(result_matrix)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    # print(accuracy)
    # plot_confusion_matrix(rf, X_test, y_test)
    # plt.title('Confusion Matrix Valiation Set - GridSearch Params')
    # plt.show()
    #

    # rf = RandomForestClassifier(criterion='entropy', max_depth=4, n_estimators=36, random_state=0)
    # rf.fit(X_train,y_train)
    # pred2 = rf.predict(X_test)
    # result_matrix = confusion_matrix(y_test, pred2)
    # print(result_matrix)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    # print(accuracy)
    # plot_confusion_matrix(rf, X_test, y_test)
    # plt.title('Confusion Matrix Valiation Set - RandomizedSearch Params')
    # plt.show()




    # accuracies.append(accuracy)

    # maxtre = confusion_matrix(pred1, pred2)
    #
    #
    #
    #
    # plt.plot(accuracies)
    # plt.show()


