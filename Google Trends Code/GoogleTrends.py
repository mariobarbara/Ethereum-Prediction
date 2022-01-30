import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from math import log2
### PARAMS FOR TESTS ###
alpha = 0.5
MAX_CLUSTERS = 25
MAX_RUNS = 30
MAX_ITER = 500
MAX_TOL = 0.001
map_ = {'ALG': None, 'CLU': None, 'RUN': None, 'ITER': None, 'TOL': None}
printRes = False
printCM = True
acc_trace = []
########################
#TODO:
# 1. greedy approach for each test
# 2. add assert for params before using it
# 3. add more info when plt the result ex : loss, ...
# 4. find a (new) way to cal the accuracy or determind the best param

# calculate mean squared error
def loss_fn( predicted,actual):
	sum_square_error = 0.0
	for i in range(len(actual)):
		sum_square_error += (actual[i] - predicted[i])**2.0
	mean_square_error = 1.0 / len(actual) * sum_square_error
	return mean_square_error

# Encode
def Encode(val):
    if val == -1:
        return [2,1,1]
    if val == 0 :
        return [1,2,1]
    if val == 1 :
        return [1,1,2]
    raise Exception("Error : Encode - unvalide label")

# calculate categorical cross entropy
def categorical_cross_entropy(actual1, predicted1):
    actual,predicted = [],[]
    assert len(actual1) == len(predicted1)
    N = len(actual1)
    for i in range(N):
        actual.append(Encode(actual1[i]))
        predicted.append(Encode(predicted[i]))

    sum_score = 0.0
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            sum_score += actual[i][j] * log2(1e-15 + predicted[i][j])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score

class K_MEANS:
    def __int__(self):
        pass
    #####
    def calc_Accuracy(self, prediction, acctual1, model,y_train):
        acctual = np.array(acctual1)
        total = 0
        accurate = 0
        assert len(prediction) == len(acctual)
        N = range(0, len(model.labels_), 1)
        for c in prediction:
            indcies = [a[0] for a in zip(N, model.labels_) if a[1] == c]
            pred = self.getPred(indcies, y_train)
            if pred == acctual[total]:
                accurate += 1
            total += 1
        return accurate / total

    def getPred(self, indcies, lables1):
        lables = np.array(lables1)
        c0, c1, c2 = 0, 0, 0
        for l in indcies:
            c0 += lables[l] == 0
            c1 += lables[l] == 1
            c2 += lables[l] == -1

        if c0 == c1 or c0 == c1 or c1 == c2:
            return 0
        res = max(c0, c1, c2)
        if res == c0:
            return 0
        if res == c1:
            return 1
        if res == c2:
            return -1
        print("ERROR")
        exit(-1)
        return -1

    def updateParam(self, val, name):
        map_[name] = val

    ## TESTS ##
    def AlgorithmTest(self, X_train, y_train, X_val, y_val):
        param_array = {'auto', 'full', 'elkan'}
        normal_accuracy_array = []
        losses = []
        best_accuracy = -1
        best_param = -1
        for algo in param_array:
            model = KMeans(algorithm=algo,random_state=42).fit(X_train,y_train)
            # Predict the classes of new, unused data
            prediction = model.predict(X_val)
            # Calc accuracy
            accuracy = self.calc_Accuracy(prediction, y_val, model, y_train)
            normal_accuracy_array.append(accuracy)
            # Calc loss
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param = algo
        self.updateParam(best_param, 'ALG')

        if printRes:
            fig = plt.figure(figsize=(10, 5))
            # creating the bar plot
            langs = list(param_array)
            students = normal_accuracy_array
            plt.bar(langs, students, color='maroon',
                    width=0.4)
            plt.xlabel('Algorithm')
            plt.ylabel('Accuracy')
            plt.show()
        acc_trace.append(best_accuracy)
        return best_param

    def ClusterNumTest(self, X_train, y_train, X_val, y_val):
        assert map_['ALG'] is not None
        param_array = range(3, MAX_CLUSTERS, 1)
        normal_accuracy_array = []
        losses = []
        best_accuracy = -1
        best_param = -1
        for n in param_array:
            model = KMeans(algorithm=map_['ALG'], n_clusters=n, random_state=42).fit(X_train, y_train)
            # Predict the classes of new, unused data
            prediction = model.predict(X_val)
            # Calc accuracy
            accuracy = self.calc_Accuracy(prediction, y_val, model, y_train)
            normal_accuracy_array.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param = n
        self.updateParam(best_param, 'CLU')
        self.printResults(param_array, normal_accuracy_array, "number of clusters - Accuracy", best_param, best_accuracy)
        acc_trace.append(best_accuracy)
        return best_param

    def AlgoRunNumTest(self,X_train,y_train,X_val,y_val):
        assert map_['CLU'] is not None
        param_array = range(1, MAX_RUNS, 1)
        normal_accuracy_array = []
        losses = []
        best_accuracy = -1
        best_param = -1
        for n in param_array:
            model = KMeans(algorithm=map_['ALG'],n_clusters=map_['CLU'],n_init=n,random_state=42).fit(X_train, y_train)
            # Predict the classes of new, unused data
            prediction = model.predict(X_val)
            # Calc accuracy
            accuracy = self.calc_Accuracy(prediction, y_val,model,y_train)
            normal_accuracy_array.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param = n
        self.updateParam(best_param ,'RUN')
        self.printResults(param_array , normal_accuracy_array,"number of runs - Accuracy",best_param,best_accuracy)

        acc_trace.append(best_accuracy)
        return best_param

    def IterNumTest(self,X_train,y_train,X_val,y_val):
        assert map_['RUN'] is not None
        param_array = range(10, MAX_ITER, 10)
        normal_accuracy_array = []
        losses = []
        best_accuracy = -1
        best_param = -1
        for n in param_array:
            model = KMeans(algorithm=map_['ALG'],n_clusters=map_['CLU'],n_init=map_['RUN'],max_iter=n,random_state=42).fit(X_train, y_train)
            # Predict the classes of new, unused data
            prediction = model.predict(X_val)
            # Calc accuracy
            accuracy = self.calc_Accuracy(prediction, y_val,model,y_train)
            normal_accuracy_array.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param = n
        self.updateParam(best_param,'ITER')
        self.printResults(param_array, normal_accuracy_array,"number of max iterations per run - Accuracy",best_param,best_accuracy)
        acc_trace.append(best_accuracy)
        return best_param

    def TolTest(self,X_train,y_train,X_val,y_val):
        assert map_['ITER'] is not None
        param_array = np.arange(-MAX_TOL, MAX_TOL, 0.0001)
        normal_accuracy_array = []
        losses = []
        best_accuracy = -1
        best_param = -1
        for n in param_array:
            model = KMeans(algorithm=map_['ALG'],n_clusters=map_['CLU'],n_init=map_['RUN'],max_iter=map_['ITER'],tol=n,random_state=42).fit(X_train, y_train)
            # Predict the classes of new, unused data
            prediction = model.predict(X_val)
            # Calc accuracy
            accuracy = self.calc_Accuracy(prediction, y_val,model,y_train)
            normal_accuracy_array.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_param = n
        self.updateParam(best_param, 'TOL')
        assert map_['TOL'] is not None
        self.printResults(param_array, normal_accuracy_array,"Relative tolerance with regards \n to Frobenius norm - Accuracy",best_param,best_accuracy)
        acc_trace.append(best_accuracy)
        return best_param

    ## Print results as a graph ##
    def printResults(self, param_values, normalCorrectP, title, best_param, best_accuracy):
        if printRes is False:
            return
        assert len(param_values) == len(normalCorrectP)
        plt.plot(param_values, normalCorrectP)
        param = title.split()
        plt.xlabel(title[:-11])
        plt.ylabel("accuracy")
        plt.title(title)
        plt.plot([best_param], [best_accuracy], 'black', markersize=5, marker='s')
        plt.show()

    def printConfusionMatrix(self,X_trainog, X_testog, y_trainog, y_testog ,name = None,model = None):
        if printCM == False:
            return
        classifier = None
        title = None
        if name is None :
            classifier = model
        else :
            if name == 'AlgorithmTest':
                title = 'Algorithm'
                classifier = KMeans(algorithm=map_['ALG'],random_state=42).fit(X_trainog, y_trainog)
            if name == 'ClusterNumTest':
                title = 'Number of Cluster'
                classifier = KMeans(algorithm=map_['ALG'], n_clusters=map_['CLU'] ,random_state=42).fit(X_trainog, y_trainog)
            if name == 'AlgoRunNumTest':
                title = 'Algorithm Runs'
                classifier = KMeans(algorithm=map_['ALG'], n_clusters=map_['CLU'], n_init=map_['RUN'] ,random_state=42).fit(X_trainog, y_trainog)
            if name == 'IterNumTest':
                title = 'Max Iteration'
                classifier = KMeans(algorithm=map_['ALG'], n_clusters=map_['CLU'], n_init=map_['RUN'], max_iter=map_['ITER'] ,random_state=42).fit(X_trainog, y_trainog)
            if name == 'TolTest':
                title = 'Tol'
                classifier = KMeans(algorithm=map_['ALG'], n_clusters=map_['CLU'], n_init=map_['RUN'], max_iter=map_['ITER'] , tol=map_['TOL'] ,random_state=42).fit(X_trainog, y_trainog)

        assert classifier is not None
        prediction = classifier.predict(X_testog)
        N = range(0, len(classifier.labels_), 1)
        yhat = []
        for c in prediction:
            indcies = [a[0] for a in zip(N, classifier.labels_) if a[1] == c]
            pred = self.getPred(indcies, y_trainog)
            yhat.append(pred)

        cm = confusion_matrix(np.array(y_testog),yhat,normalize='all')
        cmd = ConfusionMatrixDisplay(cm, display_labels=['-1', '0','1'])
        cmd.plot()
        x = 0
        #accuracy = (result_matrix[0][0] + result_matrix[1][1]) / np.sum(result_matrix,initial=0)

if __name__ == '__main__':

    all_data = pd.read_csv("google_trends_dataset_processed.xls")
    dd = pd.DataFrame(all_data)
    l1 = all_data['label1']
    for l in ['label1']:
        all_data.drop(l, axis=1, inplace=True)
    ## split data
    X_trainog, X_testog, y_trainog, y_testog = train_test_split(all_data, l1, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_trainog, y_trainog, test_size=0.25, shuffle=False)


    Tests = ['AlgorithmTest','ClusterNumTest','AlgoRunNumTest','IterNumTest','TolTest']
    best_params = []
    for test in Tests:
        m = globals()['K_MEANS']()
        func = getattr(m, test)
        res = func(np.array(X_train),np.array(y_train),np.array(X_val),np.array(y_val))
        func = getattr(m, 'printConfusionMatrix')
        func(X_trainog, X_testog, y_trainog, y_testog ,name=test,model=None)
    model = KMeans(algorithm=map_['ALG'], n_clusters=map_['CLU'], n_init=map_['RUN'], max_iter=map_['ITER'], tol=map_['TOL'],random_state=42).fit(X_trainog, y_trainog)
    # Predict the classes of new, unused data
    print(map_)
    prediction = model.predict(X_testog)
    K_MEANS().printConfusionMatrix(X_trainog, X_testog, y_trainog, y_testog ,name=None,model=model)
    acc = K_MEANS().calc_Accuracy(prediction, y_testog, model, y_trainog)
    print(acc)

