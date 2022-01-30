import matplotlib.pyplot as plt
import tweepy as tw
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from pytrends import dailydata
from scipy.stats.stats import pearsonr
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from matplotlib.pyplot import figure
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier




if __name__ == '__main__':
    all_data = pd.read_csv('dataset.csv')
    for l in ['Label1', 'Label3', 'Label5', 'Date(UTC)']:
        all_data.drop(l, axis=1, inplace=True)
    x = all_data.var()
    features = list(all_data.columns)
    variance = []
    for r in x:
        variance.append(r)

    plt.figure(figsize=(15, 10))
    plt.plot(features, variance, 'o')
    plt.xticks(rotation=90)
    plt.yscale('log')
    # plt.axhline(y=0.5, color='r', linestyle='--')
    plt.tight_layout()

    plt.show()
    print(all_data.var())