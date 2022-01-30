import numpy as np
import pandas as pd
from sklearn.metrics import  plot_confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

### Params for K-cross validation testing
alpha = 0.5
p_factor_n_leaf = 0.5
max_n_features = 15
#####################################################################

def calc_Accuracy(prediction,acctual):
    total = 0
    accurate = 0
    for pred in prediction:
        if pred == acctual[total]:
            accurate += 1
        total += 1
    return accurate/total

def calc_Accuracy_Advance(prediction, acctual):
    total = 0
    accurate = 0
    accurate_0 = 0
    for pred in prediction:
        if pred == acctual[total]:
            accurate += 1
        elif pred == 0:
            accurate_0 += 1
        total += 1
    return (accurate/total) + alpha*(accurate_0/total)

def normalID3(feat, dataset, labels, validation_p, prediction_p, train_p):
    train_features = dataset[:(train_p+validation_p)][feat]
    train_target = labels[:(train_p+validation_p)]

    test_features = dataset[(train_p+validation_p):][feat]
    test_target = labels[(train_p+validation_p):]
    # train the model
    tree = DecisionTreeClassifier(criterion='entropy').fit(train_features, train_target)
    max_depth = tree.get_depth()
    #Predict the classes of new, unused data
    prediction = tree.predict(test_features)
    #Calc accuracy
    accuracy = calc_Accuracy(prediction, test_target.array)
    print('ID3 normal accuracy ')
    print(accuracy)
    accuracy_advance = calc_Accuracy_Advance(prediction, test_target.array)
    print('ID3 advance accuracy ')
    print(accuracy_advance)

def criterion_Test(feat, dataset, labels, validation_p, prediction_p, train_p):
    train_features = dataset[:(train_p)][feat]
    train_target = labels[:(train_p)]

    test_features = dataset[(train_p):(train_p + validation_p)][feat]
    test_target = labels[(train_p):(train_p + validation_p)]
    # train the model with entropy mode
    tree1 = DecisionTreeClassifier(criterion='entropy').fit(train_features, train_target)
    prediction1 = tree1.predict(test_features)
    accuracy1 = calc_Accuracy(prediction1, test_target.array)
    accuracy_advance1 = calc_Accuracy_Advance(prediction1, test_target.array)




    # train the model with gini mode
    tree2 = DecisionTreeClassifier(criterion='gini').fit(train_features, train_target)
    prediction2 = tree2.predict(test_features)
    accuracy2 = calc_Accuracy(prediction2, test_target.array)
    accuracy_advance2 = calc_Accuracy_Advance(prediction2, test_target.array)

    print("gini")
    print(accuracy1)
    print(accuracy_advance1)
    print_ID3ConfusionMatrix(tree2,test_features,test_target,"ConfusionMatrix for criterion - gini");

def minSamplesPerLeaf_Test(feat, dataset, labels, validation_p, prediction_p, train_p):
    train_features = dataset[:(train_p)][feat]
    train_target = labels[:(train_p)]

    test_features = dataset[(train_p):(train_p + validation_p)][feat]
    test_target = labels[(train_p):(train_p + validation_p)]
    min_samples_array = range(1, 50, 2)
    normal_accuracy_array = []
    advance_accuracy_array = []
    best_accuracy = -1
    best_param = -1
    best_accuracy_advance = -1
    best_param_advance = -1
    for m in min_samples_array:
        tree = DecisionTreeClassifier(criterion='gini', min_samples_leaf=m).fit(train_features, train_target)
        #Predict the classes of new, unused data
        prediction = tree.predict(test_features)
        #Calc accuracy
        accuracy = calc_Accuracy(prediction, test_target.array)
        normal_accuracy_array.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_param = m

        accuracy_advance = calc_Accuracy_Advance(prediction, test_target.array)
        advance_accuracy_array.append(accuracy_advance)
        if accuracy_advance > best_accuracy_advance:
            best_accuracy_advance = accuracy_advance
            best_param_advance = m
    return best_param,best_param_advance
    #printResults(min_samples_array, normal_accuracy_array, advance_accuracy_array, "min samples per leaf",best_param,best_accuracy,best_param_advance,best_accuracy_advance)

def maxDepth_Test(feat, dataset, labels, validation_p, prediction_p, train_p, max_depth):
    train_features = dataset[:(train_p)][feat]
    train_target = labels[:(train_p)]

    test_features = dataset[(train_p):(train_p + validation_p)][feat]
    test_target = labels[(train_p):(train_p + validation_p)]
    max_depth_array = range(10, max_depth+5, 1)
    normal_accuracy_array = []
    advance_accuracy_array = []
    best_accuracy = -1
    best_param = -1
    best_accuracy_advance = -1
    best_param_advance = -1
    for d in max_depth_array:
        tree = DecisionTreeClassifier(criterion='gini', max_depth=d).fit(train_features, train_target)
        #Predict the classes of new, unused data
        prediction = tree.predict(test_features)
        #Calc accuracy
        accuracy = calc_Accuracy(prediction, test_target.array)
        normal_accuracy_array.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_param = d

        accuracy_advance = calc_Accuracy_Advance(prediction, test_target.array)
        advance_accuracy_array.append(accuracy_advance)
        if accuracy_advance > best_accuracy_advance:
            best_accuracy_advance = accuracy_advance
            best_param_advance = d
    #print the accuracy graphs per min samples
    return best_param,best_param_advance
    #printResults(max_depth_array, normal_accuracy_array,advance_accuracy_array, "Max depth",best_param,best_accuracy,best_param_advance,best_accuracy_advance)

def minIG_Test(feat,dataset, labels, validation_p, prediction_p, train_p):
    train_features = dataset[:(train_p)][feat]
    train_target = labels[:(train_p)]

    test_features = dataset[(train_p):(train_p + validation_p)][feat]
    test_target = labels[(train_p):(train_p + validation_p)]
    max_depth_array = np.arange(0.0, 0.02, 0.0001)
    normal_accuracy_array = []
    advance_accuracy_array = []
    best_accuracy = -1
    best_param = -1
    best_accuracy_advance = -1
    best_param_advance = -1
    for d in max_depth_array:

        tree = DecisionTreeClassifier(criterion='gini', ccp_alpha=d).fit(train_features, train_target)
        #Predict the classes of new, unused data
        prediction = tree.predict(test_features)
        #Calc accuracy
        accuracy = calc_Accuracy(prediction, test_target.array)
        normal_accuracy_array.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_param = d

        accuracy_advance = calc_Accuracy_Advance(prediction, test_target.array)
        advance_accuracy_array.append(accuracy_advance)
        if accuracy_advance > best_accuracy_advance:
            best_accuracy_advance = accuracy_advance
            best_param_advance = d
    #print the accuracy graphs per min samples
    #printResults(max_depth_array, normal_accuracy_array,advance_accuracy_array, "Minimal cost-complexity pruning",best_param,best_accuracy,best_param_advance,best_accuracy_advance)
    return best_param,best_param_advance

def maxLeafNodes_Test(feat, dataset, labels, validation_p, prediction_p, train_p, max_leaf_num):
    train_features = dataset[:(train_p)][feat]
    train_target = labels[:(train_p)]
    test_features = dataset[(train_p):(train_p + validation_p)][feat]
    test_target = labels[(train_p):(train_p + validation_p)]

    low_n_leaf = int(max_leaf_num*(1-p_factor_n_leaf))
    high_n_leaf = int(max_leaf_num * (1 + p_factor_n_leaf))
    max_leaf_array = range(low_n_leaf, high_n_leaf, 5)
    normal_accuracy_array = []
    advance_accuracy_array = []
    best_accuracy = -1
    best_param = -1
    best_accuracy_advance = -1
    best_param_advance = -1
    for m in max_leaf_array:

        tree = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=m).fit(train_features, train_target)
        #Predict the classes of new, unused data
        prediction = tree.predict(test_features)
        #Calc accuracy
        accuracy = calc_Accuracy(prediction, test_target.array)
        normal_accuracy_array.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_param = m

        accuracy_advance = calc_Accuracy_Advance(prediction, test_target.array)
        advance_accuracy_array.append(accuracy_advance)
        if accuracy_advance > best_accuracy_advance:
            best_accuracy_advance = accuracy_advance
            best_param_advance = m
    #print the accuracy graphs per max leafs
    #printResults(max_leaf_array, normal_accuracy_array, advance_accuracy_array, "Max Leaf Nodes",best_param,best_accuracy,best_param_advance,best_accuracy_advance)
    return best_param,best_param_advance

def maxNumFeatures_Test(feat, dataset, labels, validation_p, prediction_p, train_p):
    train_features = dataset[:(train_p)][feat]
    train_target = labels[:(train_p)]
    test_features = dataset[(train_p):(train_p + validation_p)][feat]
    test_target = labels[(train_p):(train_p + validation_p)]

    max_features_array = range(5, max_n_features, 1)
    normal_accuracy_array = []
    advance_accuracy_array = []
    best_accuracy = -1
    best_param = -1
    best_accuracy_advance = -1
    best_param_advance = -1
    for m in max_features_array:
        tree = DecisionTreeClassifier(criterion='gini', max_features=m).fit(train_features, train_target)
        #Predict the classes of new, unused data
        prediction = tree.predict(test_features)
        #Calc accuracy
        accuracy = calc_Accuracy(prediction, test_target.array)
        normal_accuracy_array.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_param = m

        accuracy_advance = calc_Accuracy_Advance(prediction, test_target.array)
        advance_accuracy_array.append(accuracy_advance)
        if accuracy_advance > best_accuracy_advance:
            best_accuracy_advance = accuracy_advance
            best_param_advance = m
    #print the accuracy graphs per max features to split
    printResults(max_features_array, normal_accuracy_array, advance_accuracy_array, "Max Features to Split",best_param,best_accuracy,best_param_advance,best_accuracy_advance)
    return best_param,best_param_advance


def greedySearch(validation_features,validation_target ,trainX_features,trainX_target):
    names = ["min samples per leaf", "max depth", "min IG", "max leaf nodes", "max features"]
    nulls = [None, None, None, None, None]
    names_array = []
    accuracy_array = []
    accuracy_array_advance = []
    vals = [50, 25, 0.001, 385, 14]
    vals_advance = [3, 25, 0.00011, 385, 7]
    tree0 = DecisionTreeClassifier(criterion='gini').fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree0.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree0.predict(validation_features), validation_target.array))
    names_array.append("default - gini")
    ###########
    tree1 = DecisionTreeClassifier(criterion='gini',min_samples_leaf=50).fit(trainX_features, trainX_target)
    tree2 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree1.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree2.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf")
    ###########
    tree3 = DecisionTreeClassifier(criterion='gini',max_depth=25).fit(trainX_features, trainX_target)
    tree4 = DecisionTreeClassifier(criterion='gini', max_depth=25).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree3.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree4.predict(validation_features), validation_target.array))
    names_array.append("max depth")
    ###########
    tree5 = DecisionTreeClassifier(criterion='gini',ccp_alpha=0.001).fit(trainX_features, trainX_target)
    tree6 = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.00011).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree5.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree6.predict(validation_features), validation_target.array))
    names_array.append("min IG")
    ###########
    tree7 = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=385).fit(trainX_features, trainX_target)
    tree8 = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=385).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree7.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree8.predict(validation_features), validation_target.array))
    names_array.append("max leaf node")
    ###########
    tree9 = DecisionTreeClassifier(criterion='gini',max_features=14).fit(trainX_features, trainX_target)
    tree10 = DecisionTreeClassifier(criterion='gini', max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree9.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree10.predict(validation_features), validation_target.array))
    names_array.append("max features")
    ###########
    tree11 = DecisionTreeClassifier(criterion='gini',min_samples_leaf=50,max_depth=25).fit(trainX_features, trainX_target)
    tree12 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,max_depth=25).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree11.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree12.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max depth")
    ###########
    tree13 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50, ccp_alpha=0.001).fit(trainX_features, trainX_target)
    tree14 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50, ccp_alpha=0.00011).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree13.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree14.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,min IG")
    ###########
    tree15 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50, max_leaf_nodes=385).fit(trainX_features, trainX_target)
    tree16 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50, max_leaf_nodes=385).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree15.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree16.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max leaf node")
    ###########
    tree17 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50, max_features=14).fit(trainX_features, trainX_target)
    tree18 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50, max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree17.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree18.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max features")
    ###########
    tree19 = DecisionTreeClassifier(criterion='gini', max_depth=25 , ccp_alpha=0.001).fit(trainX_features, trainX_target)
    tree20 = DecisionTreeClassifier(criterion='gini', max_depth=25 , ccp_alpha=0.00011).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree19.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree20.predict(validation_features), validation_target.array))
    names_array.append("max depth ,min IG")
    ###########
    tree21 = DecisionTreeClassifier(criterion='gini', max_depth=25 , max_leaf_nodes=385).fit(trainX_features, trainX_target)
    tree22 = DecisionTreeClassifier(criterion='gini', max_depth=25 , max_leaf_nodes=385).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree21.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree22.predict(validation_features), validation_target.array))
    names_array.append("max depth ,max leaf node")
    ###########
    tree23 = DecisionTreeClassifier(criterion='gini', max_depth=25, max_features=14).fit(trainX_features, trainX_target)
    tree24 = DecisionTreeClassifier(criterion='gini', max_depth=25, max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree23.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree24.predict(validation_features), validation_target.array))
    names_array.append("max depth ,max features")
    ###########
    tree25 = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.001 , max_leaf_nodes=385).fit(trainX_features, trainX_target)
    tree26 = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.00011 , max_leaf_nodes=385).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree25.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree26.predict(validation_features), validation_target.array))
    names_array.append("min IG ,max leaf node")
    ###########
    tree27 = DecisionTreeClassifier(criterion='gini',ccp_alpha=0.001 , max_features=14).fit(trainX_features, trainX_target)
    tree28 = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.001 , max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree27.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree28.predict(validation_features), validation_target.array))
    names_array.append("min IG ,max features")
    ###########
    tree29 = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=385 ,max_features=14).fit(trainX_features, trainX_target)
    tree30 = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=385 ,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree29.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree30.predict(validation_features), validation_target.array))
    names_array.append("max leaf node ,max features")
    ###########
    tree31 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,max_depth=25,ccp_alpha=0.001).fit(trainX_features, trainX_target)
    tree32 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,max_depth=25,ccp_alpha=0.00011).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree31.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree32.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max depth ,min IG")
    ###########
    tree33 = DecisionTreeClassifier(criterion='gini',  min_samples_leaf=50,max_depth=25,max_leaf_nodes=385).fit(trainX_features, trainX_target)
    tree34 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,max_depth=25,max_leaf_nodes=385).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree33.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree34.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max depth ,max leaf node")
    ###########
    tree35 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50 ,max_depth=25,max_features=14).fit(trainX_features, trainX_target)
    tree36 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3 ,max_depth=25,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree35.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree36.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max depth ,max features")
    ###########
    tree37 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,ccp_alpha=0.001 ,max_leaf_nodes=385).fit(trainX_features, trainX_target)
    tree38 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,ccp_alpha=0.00011 ,max_leaf_nodes=385).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree37.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree38.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,min IG ,max leaf node")
    ###########
    tree39 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,ccp_alpha=0.001 ,max_features=14).fit(trainX_features, trainX_target)
    tree40 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,ccp_alpha=0.00011 ,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree39.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree40.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,min IG ,max features")
    ###########
    tree41 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,max_leaf_nodes=385,max_features=14).fit(trainX_features, trainX_target)
    tree42 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,max_leaf_nodes=385,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree41.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree42.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max leaf node ,max features")
    ###########
    tree43 = DecisionTreeClassifier(criterion='gini', max_depth=25, ccp_alpha=0.001, max_leaf_nodes=385).fit(trainX_features, trainX_target)
    tree44 = DecisionTreeClassifier(criterion='gini', max_depth=25, ccp_alpha=0.00011, max_leaf_nodes=385).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree43.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree44.predict(validation_features), validation_target.array))
    names_array.append("max depth ,min IG ,max leaf node")
    ###########
    tree45 = DecisionTreeClassifier(criterion='gini', max_depth=25, ccp_alpha=0.001, max_features=14).fit(trainX_features, trainX_target)
    tree46 = DecisionTreeClassifier(criterion='gini', max_depth=25, ccp_alpha=0.00011, max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree45.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree46.predict(validation_features), validation_target.array))
    names_array.append("max depth ,min IG ,max features")
    ###########
    tree47 = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.001, max_leaf_nodes=385 ,max_features=14).fit(trainX_features, trainX_target)
    tree48 = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.00011, max_leaf_nodes=385 ,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree47.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree48.predict(validation_features), validation_target.array))
    names_array.append("min IG ,max leaf node ,max features")
    ###########
    tree49 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,max_depth=25,ccp_alpha=0.001,max_leaf_nodes=385).fit(trainX_features, trainX_target)
    tree50 = DecisionTreeClassifier(criterion='gini',  min_samples_leaf=3,max_depth=25,ccp_alpha=0.00011,max_leaf_nodes=385).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree49.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree50.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max depth ,min IG ,max leaf node")
    ###########
    tree51 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,max_depth=25,ccp_alpha=0.001,max_features=14).fit(trainX_features, trainX_target)
    tree52 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,max_depth=25,ccp_alpha=0.00011,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree51.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree52.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max depth ,min IG ,max features")
    ###########
    tree53 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,max_depth=25,max_leaf_nodes=385,max_features= 14).fit(trainX_features, trainX_target)
    tree54 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,max_depth=25,max_leaf_nodes=385,max_features= 7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree53.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree54.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max depth ,max leaf node ,max features")
    ###########
    tree55 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,ccp_alpha=0.001,max_leaf_nodes=385,max_features=14).fit(trainX_features, trainX_target)
    tree56 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,ccp_alpha=0.00011,max_leaf_nodes=385,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree55.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree56.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,min IG ,max leaf node ,max features")
    ###########
    tree57 = DecisionTreeClassifier(criterion='gini',min_samples_leaf=50, max_depth=25,ccp_alpha=0.001,max_leaf_nodes=385,max_features=14).fit(trainX_features, trainX_target)
    tree58 = DecisionTreeClassifier(criterion='gini',min_samples_leaf=3,max_depth=25,ccp_alpha=0.00011,max_leaf_nodes=385,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree57.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree58.predict(validation_features), validation_target.array))
    names_array.append("max depth ,min IG ,max leaf node ,max features")
    ###########
    tree59 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,max_depth=25,ccp_alpha=0.001 ,max_leaf_nodes=385,max_features=14).fit(trainX_features, trainX_target)
    tree60 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,max_depth=25,ccp_alpha=0.00011 ,max_leaf_nodes=385,max_features=7).fit(trainX_features, trainX_target)
    accuracy_array.append(calc_Accuracy(tree59.predict(validation_features), validation_target.array))
    accuracy_array_advance.append(calc_Accuracy_Advance(tree60.predict(validation_features), validation_target.array))
    names_array.append("min samples per leaf ,max depth ,min IG ,max leaf node ,max features")
    ######################
    # PLOT THE RESULTS
    # fig = plt.figure(figsize=(10,10))
    # plt.plot(names_array,accuracy_array, 'o', color='b')
    # plt.plot(names_array,accuracy_array_advance, 'o', color='r')
    # plt.legend(["Normal calc", "Weighted Calc"])
    # plt.title("Greedy Search")
    # plt.xticks(rotation=90)
    # fig.canvas.draw()
    # plt.tight_layout()
    # plt.show()

########################################################################
def printResults(pram_values, normalCorrectP,advanceCorrectP, title,best_param,best_accuracy,best_param_advance,best_accuracy_advance):
    plt.plot(pram_values,normalCorrectP,'b')
    plt.plot(pram_values,advanceCorrectP,'r')
    plt.legend(["Normal accuracy calculation", "Weighted accuracy calculation"])
    plt.title(title)
    plt.plot([best_param],[best_accuracy],'black',markersize=5,marker='s')
    plt.plot([best_param_advance], [best_accuracy_advance], 'black', markersize=5, marker='s')
    plt.show()

def print_ID3ConfusionMatrix(classifier,X_test, y_test,name):
    plot_confusion_matrix(classifier,X_test,y_test)
    plt.title(name)
    plt.show()
    # titles_options = [
    #     ("Confusion matrix, without normalization", None),
    # ]
    # for title, normalize in titles_options:
    #     ConfusionMatrixDisplay.
    #     disp = ConfusionMatrixDisplay.from_predictions(
    #         classifier,
    #         X_test,
    #         y_test,
    #         display_labels=["Class -1", "Class 0", "Class 1"],
    #         cmap=plt.cm.Blues,
    #         normalize=normalize,
    #     )
    #     disp.ax_.set_title(title)
    #
    #     print(title)
    #     print(disp.confusion_matrix)
    #
    # plt.show()

if __name__ == '__main__':
    dataset = pd.read_csv("dataset.csv")
    l1 = dataset['Label1']
    l3 = dataset['Label3']
    l5 = dataset['Label5']
    labels = l1
    #feat = ['open', 'Volume USDT', 'close_SMA_50', 'open_SMA_50', 'high_SMA_50', 'low_WMA_10', 'atr_ema_based','AverageGasPrice_avg']
    feat = [
            'Volume USDT', 'AverageBlockTime_avg', 'AverageDailyTransactionFee_avg',
            'AverageDailyTransactionFee_SMA_10', 'UniqueAddressesCount_avg', 'DailyBlockRewards_avg',
            'DailyTransactions_SMA_50','low_WMA_50', 'DailyTransactions_SMA_10','close_WMA_50',
            'AverageDailyTransactionFee_SMA_50' , 'NetworkHashrate_SMA_10', 'close_WMA_20', 'NetworkHashrate_avg',
            'DailyVerifiedContracts _avg']

    for l in list(dataset.columns):
        if l not in feat:
            dataset.drop(l, axis=1, inplace=True)
    num_features = len(dataset) #1314
    validation_p = int(num_features*0.2) #262
    prediction_p = validation_p #262
    train_p = num_features - validation_p - prediction_p #790
    train_features = dataset[:(train_p+validation_p)][feat]
    train_target = labels[:(train_p+validation_p)]

    test_features = dataset[(train_p+validation_p):][feat]
    test_target = labels[(train_p+validation_p):]
    # train the model
    # tree = DecisionTreeClassifier().fit(train_features, train_target)
    # print_ID3ConfusionMatrix(tree, test_features, test_target, "ConfusionMatrix for default params")
    # acc1 = calc_Accuracy(tree.predict(test_features),test_target.array)
    # print(acc1)
    # acc2 = calc_Accuracy_Advance(tree.predict(test_features),test_target.array)
    # print(acc2)

    ############# K cross validation Testing #############

    # ------------ gini OR entropy mode ------------
    #criterion_Test(feat, dataset, labels, validation_p, prediction_p, train_p)

    # ------------ min samples leaf ------------
    #minSamplesPerLeaf_Test(feat, dataset, labels, validation_p, prediction_p, train_p)
    #tree1 = DecisionTreeClassifier(criterion='gini',min_samples_leaf=x1).fit(train_features, train_target)
    #accuracy1 = calc_Accuracy(tree1.predict(test_features),test_target.array)
    #print(accuracy1)
    #tree2 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=x2).fit(train_features, train_target)
    #accuracy2 = calc_Accuracy_Advance(tree2.predict(test_features),test_target.array)
    #print(accuracy2)

    # ------------ max depth ------------
    #x1,x2 = maxDepth_Test(feat, dataset, labels, validation_p, prediction_p, train_p, 30)
    #tree1 = DecisionTreeClassifier(criterion='gini',max_depth=x1).fit(train_features, train_target)
    #print_ID3ConfusionMatrix(tree1, test_features, test_target, "ConfusionMatrix for max Depth - normal calc")
    #accuracy1 = calc_Accuracy(tree1.predict(test_features),test_target.array)
    #print(accuracy1)
    #tree2 = DecisionTreeClassifier(criterion='gini', max_depth=x2).fit(train_features, train_target)
    #print_ID3ConfusionMatrix(tree2, test_features, test_target, "ConfusionMatrix for max Depth - weighted calc")
    #accuracy2 = calc_Accuracy_Advance(tree2.predict(test_features),test_target.array)
    #print(accuracy2)
    # ------------ min IG - cc alpha ------------
    #minIG_Test(feat, dataset, labels, validation_p, prediction_p, train_p)
    #x1,x2 = minIG_Test(feat, dataset, labels, validation_p, prediction_p, train_p)
    #tree1 = DecisionTreeClassifier(criterion='gini', ccp_alpha=x1).fit(train_features, train_target)
    #print_ID3ConfusionMatrix(tree1, test_features, test_target, "ConfusionMatrix for Min IG - normal calc")
    #accuracy1 = calc_Accuracy(tree1.predict(test_features),test_target.array)
    #print(accuracy1)
    #tree2 = DecisionTreeClassifier(criterion='gini', ccp_alpha=x2).fit(train_features, train_target)
    #print_ID3ConfusionMatrix(tree2, test_features, test_target, "ConfusionMatrix for Min IG  - weighted calc")
    #accuracy2 = calc_Accuracy_Advance(tree2.predict(test_features),test_target.array)
    #print(accuracy2)
    # ------------ max leaf nodes ------------

    #x1, x2 = maxLeafNodes_Test(feat, dataset, labels, validation_p, prediction_p, train_p, 400)
    #tree1 = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=x1).fit(train_features, train_target)
    #print_ID3ConfusionMatrix(tree1, test_features, test_target, "ConfusionMatrix for Max leaf nodes - normal calc")
    #accuracy1 = calc_Accuracy(tree1.predict(test_features), test_target.array)
    #print(accuracy1)
    #tree2 = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=x2).fit(train_features, train_target)
    #print_ID3ConfusionMatrix(tree2, test_features, test_target, "ConfusionMatrix for Max leaf nodes - weighted calc")
    #accuracy2 = calc_Accuracy_Advance(tree2.predict(test_features),test_target.array)
    #print(accuracy2)
    # ------------ max features to consider for split ------------
    #x1,x2 = maxNumFeatures_Test(feat, dataset, labels, validation_p, prediction_p, train_p)
    #tree1 = DecisionTreeClassifier(criterion='gini', max_features=x1).fit(train_features, train_target)
    #print_ID3ConfusionMatrix(tree1, test_features, test_target, "ConfusionMatrix for Max features - normal calc")
    #accuracy1 = calc_Accuracy(tree1.predict(test_features), test_target.array)
    #print(accuracy1)
    #tree2 = DecisionTreeClassifier(criterion='gini', max_features=x2).fit(train_features, train_target)
    #print_ID3ConfusionMatrix(tree2, test_features, test_target, "ConfusionMatrix for Max features - weighted calc")
    #accuracy2 = calc_Accuracy_Advance(tree2.predict(test_features),test_target.array)
    #print(accuracy2)
    # ------------ Greedy Search ------------
    # trainX_features = dataset[:(train_p)][feat]
    # trainX_target = labels[:(train_p)]
    # validation_features = dataset[(train_p):(train_p + validation_p)][feat]
    # validation_target = labels[(train_p):(train_p + validation_p)]
    #greedySearch(validation_features, validation_target, trainX_features, trainX_target)
    # tree1 = DecisionTreeClassifier(criterion='gini',min_samples_leaf=50, max_depth=25, max_features=14).fit(train_features, train_target)
    # accuracy1 = calc_Accuracy(tree1.predict(test_features),test_target.array)
    # print(accuracy1)
    # tree2 = DecisionTreeClassifier(criterion='gini',max_depth=25, ccp_alpha=0.00011, max_features=7).fit(train_features, train_target)
    # accuracy2 = calc_Accuracy_Advance(tree2.predict(test_features),test_target.array)
    # print(accuracy2)
    tree0 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50,max_depth=25,max_leaf_nodes=385,max_features=14).fit(train_features, train_target)
    acc = calc_Accuracy(tree0.predict(test_features),test_target.array)
    print_ID3ConfusionMatrix(tree0, test_features, test_target, "ConfusionMatrix for best Classifier - Normal Accuracy")

    tree1 = DecisionTreeClassifier(criterion='gini', min_samples_leaf=3,max_depth=25,max_features=7).fit(train_features, train_target)
    print_ID3ConfusionMatrix(tree1, test_features, test_target, "ConfusionMatrix for best Classifier - Weighted Accuracy")
    acc_advance = calc_Accuracy_Advance(tree1.predict(test_features),test_target.array)


