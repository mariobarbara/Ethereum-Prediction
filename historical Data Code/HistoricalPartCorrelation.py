import matplotlib.pyplot as plt
import tweepy as tw
import pandas as pd
import numpy as np
import seaborn as sn
from pytrends.request import TrendReq
from pytrends import dailydata
from scipy.stats.stats import pearsonr
import datetime
from sklearn.feature_selection import VarianceThreshold


if __name__ == '__main__':
    all_data = pd.read_csv('dataset.csv')
    # list_of_column_names = list(all_data.columns)

    # features = []
    # correlations = []
    # print(list_of_column_names)
    # for i in range(1, len(list_of_column_names)-3):
    #     if list_of_column_names[i] == 'close':
    #         continue
    #     features.append(list_of_column_names[i])
    #     correlations.append(pearsonr(all_data[list_of_column_names[i]], all_data['close'])[0])
    #
    # results = []
    # # Without the Date (first col) and without Labels (last 3 cols)
    # for i in range(1, len(list_of_column_names)-3):
    #     curr = []
    #     for j in range(1, len(list_of_column_names)-3):
    #         curr.append(pearsonr(all_data[list_of_column_names[i]], all_data[list_of_column_names[j]])[0])
    #     results.append(curr)

    for l in ['Label1', 'Label3', 'Label5', 'Date(UTC)']:
        all_data.drop(l, axis=1, inplace=True)

    x = all_data.var()
    features = list(all_data.columns)
    variance = []

    for l in features:
        if x.at[l] < 0.5:
            all_data.drop(l, axis=1, inplace=True)

    # sel = VarianceThreshold(threshold=20)
    # dataset = sel.fit_transform(all_data)

    all_data.to_csv("dataset_after_variance.csv")

    fig_dims = (16, 8)
    fig, ax = plt.subplots(figsize=fig_dims)
    sn.heatmap(all_data.corr(), ax=ax)
    plt.tight_layout()
    plt.show()
    # H = np.array(results)
    # fig = plt.figure(figsize=(50, 50))
    #
    # # plt.rcParams["figure.figsize"] = [16, 9]
    #
    # ax = fig.add_subplot(111)
    # ax.set_title('Correlation between Features')
    # plt.imshow(H)
    # ax.set_aspect('equal')
    # ax.set_xticklabels(list_of_column_names)
    # ax.set_yticklabels(list_of_column_names)
    #
    # ax.set_xticks(np.arange(len(list_of_column_names)))
    # ax.set_yticks(np.arange(len(list_of_column_names)))
    #
    # plt.xticks(rotation=90)
    #
    # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    # cax.get_xaxis().set_visible(False)
    # cax.get_yaxis().set_visible(False)
    # cax.patch.set_alpha(0)
    # cax.set_frame_on(False)
    # plt.colorbar(orientation='vertical')
    # plt.show()

    # size = len(features)
    # half = len(features)*0.5
    #
    # plt.figure(figsize=(50, 50))
    # plt.plot(features[0:int(half)], correlations[0:int(half)], 'o')
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.axhline(y=0.7, color='r', linestyle='--')
    # # plt.autoscale()
    # plt.show()
    #
    # plt.figure(figsize=(50, 50))
    # plt.plot(features[int(half):size], correlations[int(half):size], 'o')
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.axhline(y=0.7, color='r', linestyle='--')
    # # plt.autoscale()
    # plt.show()
