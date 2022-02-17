import matplotlib.pyplot as plt
import tweepy as tw
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from pytrends.request import TrendReq
from pytrends import dailydata
from scipy.stats.stats import pearsonr
import datetime
from sklearn.feature_selection import VarianceThreshold


if __name__ == '__main__':
    all_data = pd.read_csv('google_trends_dataset.csv')

    # for l in ['label1']:
    #     all_data.drop(l, axis=1, inplace=True)
    # list_of_column_names = list(all_data.columns)
    #
    # x = all_data.var()
    # features = list(all_data.columns)
    # variance = []
    # for r in x:
    #     variance.append(r)
    #
    # plt.figure(figsize=(15, 10))
    # plt.plot(features, variance, 'o')
    # plt.xticks(rotation=90)
    # plt.yscale('log')
    # plt.axhline(y=1, color='r', linestyle='--')
    # plt.tight_layout()
    # plt.show()
    #
    # fig_dims = (16, 16)
    # fig, ax = plt.subplots(figsize=fig_dims)
    # sn.heatmap(all_data.corr(), ax=ax)
    # plt.tight_layout()
    # plt.show()

    # correlation = all_data.corr()
    # list_of_column_names = list(all_data.columns)
    # for l1 in list_of_column_names:
    #     for l2 in list_of_column_names:
    #         if l1 == l2 or l1 == 'label1' or l2 == 'label1':
    #             continue
    #         if correlation.at[l1, l2] > 0.7 or correlation.at[l1, l2] < -0.7:
    #             sns.pairplot(all_data, palette="bright", x_vars=[l1], y_vars=[l2],
    #                          hue='label1', kind='scatter')
    #             path = './correlationExp/' + l1 + '_' + l2
    #             plt.savefig(path)
                # plt.show()

    # features to remove according to correlation experiment
    features_to_remove = ['ether_normalized_max', 'ether_normalized_l2', 'ether_normalized_l1', 'ethereum_WMA_50', 'IQR_last_50'
                          , 'over_10_last_50', 'ethereum_WMA_20', 'ethereum_WMA_14']
    for feature in features_to_remove:
        all_data.drop(feature, axis=1, inplace=True)

    # features to remove according to variance experiment
    features_to_remove = ['Year', 'between_quantiles_30', 'between_quantiles_50', 'diff_from_median_14',
                          'diff_from_median_30']
    for feature in features_to_remove:
        all_data.drop(feature, axis=1, inplace=True)
    all_data.to_csv('google_trends_dataset_processed.csv')



