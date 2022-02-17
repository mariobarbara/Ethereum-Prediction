import matplotlib.pyplot as plt
import tweepy as tw
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sn

from pytrends.request import TrendReq
from pytrends import dailydata
from scipy.stats.stats import pearsonr
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Gets a date DD/MM/YYYY and returns each element alone
# returns day,month,year and a string "day month year"
def read_date(date):
    day, month, year = date[0:2], date[3:5], date[6:10]
    full_date = day + " " + month + " " + year
    return day, month, year, full_date

if __name__ == '__main__':
    dataset = pd.read_csv("ethereumGoogleTrends.csv")

    # Adds new columns to the dataset
    added_columns = ['label1', 'ethereum_prev', 'ethereum_unscaled_prev',
                     'ethereum_SMA_10', 'ethereum_SMA_14', 'ethereum_SMA_20', 'ethereum_SMA_50',
                     'ethereum_WMA_10', 'ethereum_WMA_14', 'ethereum_WMA_20', 'ethereum_WMA_50',
                     'ethereum_max_10', 'ethereum_max_14', 'ethereum_max_20', 'ethereum_max_50',
                     'ethereum_min_10', 'ethereum_min_14', 'ethereum_min_20', 'ethereum_min_50',
                     'Day', 'Month', 'Year', 'Weekday',
                     'over_2_last_10', 'over_2_last_14', 'over_2_last_20', 'over_2_last_50',
                     'over_5_last_10', 'over_5_last_14', 'over_5_last_20', 'over_5_last_50',
                     'over_10_last_10', 'over_10_last_14', 'over_10_last_20', 'over_10_last_50',
                     'over_15_last_10', 'over_15_last_14', 'over_15_last_20', 'over_15_last_50',
                     'median_last_30', 'median_last_14', 'var_last_30', 'var_last_14', 'range_last_14',
                     'range_last_30', 'Q1_last_30', 'Q1_last_50', 'Q3_last_30', 'Q3_last_50', 'IQR_last_30',
                     'IQR_last_50', 'over_median_14', 'over_median_30', 'between_quantiles_30', 'between_quantiles_50',
                     'diff_from_median_14', 'diff_from_median_30'
                     ]

    for col in added_columns:
        dataset[col] = ""

    rowsNum = len(dataset.index)
    for i in range(rowsNum - 1, -1, -1):
        open_price = dataset.at[i, 'open']
        close_price = dataset.at[i, 'close']
        percent_up = 1 + 1 / 100
        percent_down = 1 - 1 / 100
        if open_price * percent_down <= close_price <= open_price * percent_up:
            dataset.at[i, 'label1'] = 0
        elif open_price < close_price:
            dataset.at[i, 'label1'] = 1
        else:
            dataset.at[i, 'label1'] = -1

    days = [10, 14, 20, 50]
    # Calculating Simple Moving Average (SMA).
    for d in days:
        sma_col = 'ethereum_SMA_' + str(d)
        wma_col = 'ethereum_WMA_' + str(d)
        max_col = 'ethereum_max_' + str(d)
        min_col = 'ethereum_min_' + str(d)
        for i in range(rowsNum - 1, 51, -1):
            sum_ethereum_sma = 0
            sum_ethereum_wma = 0
            max_ethereum = 0
            min_ethereum = 101
            for j in range(1, d + 1):
                sum_ethereum_sma = sum_ethereum_sma + dataset.at[i - j, 'ethereum']
                sum_ethereum_wma = sum_ethereum_wma + dataset.at[i - j, 'ethereum'] * (d - j + 1)
                if dataset.at[i - j, 'ethereum'] > max_ethereum:
                    max_ethereum = dataset.at[i - j, 'ethereum']
                if dataset.at[i - j, 'ethereum'] < min_ethereum:
                    min_ethereum = dataset.at[i - j, 'ethereum']
            dataset.at[i, sma_col] = sum_ethereum_sma / d
            dataset.at[i, wma_col] = sum_ethereum_wma / (d * (d + 1) / 2)
            dataset.at[i, max_col] = max_ethereum
            dataset.at[i, min_col] = min_ethereum

    for d in days:
        for score in [2, 5, 10, 15]:
            for i in range(rowsNum - 1, 51, -1):
                counter = 0
                for j in range(1, d + 1):
                    if dataset.at[i - j, 'ethereum'] > score:
                        counter = counter + 1
                col_name = 'over_' + str(score) + '_last_' + str(d)
                dataset.at[i, col_name] = counter

    dataset['log_value'] = np.log(dataset['ethereum'])
    plt.plot(dataset['log_value'])
    plt.xlabel('Days (Numbered Sequentially)')
    plt.ylabel('Log Value of GT Score')
    plt.show()
    dataset['log_value_prev'] = ""
    for i in range(rowsNum - 1, 0, -1):
        dataset.at[i, 'ethereum_prev'] = dataset.at[i-1, 'ethereum']
        dataset.at[i, 'ethereum_unscaled_prev'] = dataset.at[i-1, 'ethereum_unscaled']
        dataset.at[i, 'log_value_prev'] = dataset.at[i-1, 'log_value']
    rows_to_drop = [0, 100, 300, 500, 700, 900]
    for num in rows_to_drop:
        data = dataset.iloc[num:]

        # Ranges Histogram
        ranges = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']
        number = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(rowsNum - 1, num-1 , -1):
            score = data.at[i, 'ethereum']
            if score == 100:
                number[9] = number[9] + 1
                continue
            pos = int(score / 10)
            number[pos] = number[pos] + 1
        plt.plot(ranges, number, 'o')
        plt.show()

    # Transfers all needed values from dataset into all_data
    for i in range(rowsNum-1, -1, -1):
        dataset.at[i, 'Day'], dataset.at[i, 'Month'], dataset.at[i, 'Year'], full_date = read_date(dataset.at[i, 'date'])
        weekday = datetime.datetime.strptime(full_date, '%d %m %Y').weekday()
        dataset.at[i, 'Weekday'] = weekday

    for i in range(rowsNum-1, 32, -1):
        last_14_days = dataset['ethereum'][i-14:i]
        last_30_days = dataset['ethereum'][i-30:i]
        last_50_days = dataset['ethereum'][i-50:i]
        dataset.at[i, 'median_last_30'] = last_30_days.median()
        dataset.at[i, 'median_last_14'] = last_14_days.median()
        dataset.at[i, 'var_last_30'] = last_30_days.var(axis=0)
        dataset.at[i, 'var_last_14'] = last_14_days.var(axis=0)
        dataset.at[i, 'range_last_14'] = last_14_days.max() - last_14_days.min()
        dataset.at[i, 'range_last_30'] = last_30_days.max() - last_30_days.min()
        dataset.at[i, 'Q1_last_30'] = last_30_days.quantile(0.25)
        dataset.at[i, 'Q1_last_50'] = last_50_days.quantile(0.25)
        dataset.at[i, 'Q3_last_30'] = last_30_days.quantile(0.75)
        dataset.at[i, 'Q3_last_50'] = last_50_days.quantile(0.75)
        dataset.at[i, 'IQR_last_30'] = dataset.at[i, 'Q3_last_30'] - dataset.at[i, 'Q1_last_30']
        dataset.at[i, 'IQR_last_50'] = dataset.at[i, 'Q3_last_50'] - dataset.at[i, 'Q1_last_50']
        if dataset.at[i, 'ethereum_prev'] > dataset.at[i, 'median_last_14']:
            dataset.at[i, 'over_median_14'] = 1
        else:
            dataset.at[i, 'over_median_14'] = 0
        if dataset.at[i, 'ethereum_prev'] > dataset.at[i, 'median_last_30']:
            dataset.at[i, 'over_median_30'] = 1
        else:
            dataset.at[i, 'over_median_30'] = 0
        if dataset.at[i, 'Q1_last_30'] < dataset.at[i, 'ethereum_prev'] < dataset.at[i, 'Q3_last_30']:
            dataset.at[i, 'between_quantiles_30'] = 1
        else:
            dataset.at[i, 'between_quantiles_30'] = 0
        if dataset.at[i, 'Q1_last_50'] < dataset.at[i, 'ethereum_prev'] < dataset.at[i, 'Q3_last_50']:
            dataset.at[i, 'between_quantiles_50'] = 1
        else:
            dataset.at[i, 'between_quantiles_50'] = 0
        dataset.at[i, 'diff_from_median_14'] = dataset.at[i, 'ethereum_prev'] - dataset.at[i, 'median_last_14']
        dataset.at[i, 'diff_from_median_30'] = dataset.at[i, 'ethereum_prev'] - dataset.at[i, 'median_last_30']


    # Sequential google trends scores plot
    rows_to_drop = [0,100, 300, 500, 700, 900, 1100, 1300]
    for num in rows_to_drop:
        data = dataset.iloc[num:]
        plt.plot(data['ethereum'])
        plt.title('Google Trends Scores Graph - without first ' + str(num) + ' days')
        plt.xlabel('Days (Numbered sequentially)')
        plt.ylabel('Google Trend Score')
        plt.show()

    dataset.drop("date", axis=1, inplace=True)
    dataset = dataset.iloc[700:]
    l1 = dataset['label1']
    l1 = l1.astype('int')
    # dataset.drop("label1", axis=1, inplace=True)

    plt.hist(dataset['ethereum'], bins=100)
    plt.ylabel('Frequency')
    plt.xlabel('Google Trend Score (word : Ethereum)')
    plt.show()
    dataset['ether_normalized_l1'] = preprocessing.normalize([dataset['ethereum_prev']], norm='l1')[0]
    dataset['ether_normalized_l2'] = preprocessing.normalize([dataset['ethereum_prev']], norm='l2')[0]
    dataset['ether_normalized_max'] = preprocessing.normalize([dataset['ethereum_prev']], norm='max')[0]

    dataset.to_csv('google_trends_dataset.csv')


    # options = [3, 6, 9, 12]
    # for opt1 in options:
    #     for opt2 in options:
    #         for opt3 in options:
    # clf = MLPClassifier(solver='lbfgs', activation='logistic', max_iter=10000, alpha=1e-5, hidden_layer_sizes=(12, 9, 3), random_state=1, shuffle=False)
    # curr, X_test, curr_y, y_test = train_test_split(dataset, l1, test_size=0.2, shuffle=False)
    #              X_train, X_val, y_train, y_val = train_test_split(curr, curr_y, test_size=0.25, shuffle=False)
    # clf.fit(curr, curr_y)
    # pred = clf.predict(X_test)
    # result_matrix = confusion_matrix(y_test, pred)
    # accuracy = (result_matrix[0][0] + result_matrix[1][1] + result_matrix[2][2]) / np.sum(result_matrix, initial=0)
    # # print(str(opt1) + '    ' + str(opt2) + '    ' + str(opt3))
    # print(accuracy)


