import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import datetime
from datetime import timedelta


def experiment(values, colName, title, xlabel, ylabel):
    start_date = datetime.datetime(2018, 8, 17)
    curr_date = datetime.datetime(2018, 8, 17)
    end_data = datetime.datetime(2020, 12, 31)
    strings = []
    length = len(values)
    for i in range(0, length-1):
        string = str(values[i]) + '-' + str(values[i+1])
        strings.append(string)
    strings.append('Over ' + str(values[length-1]))
    histogram = [0 for i in range(0, length)]
    while curr_date != end_data:
        filename = './tweetsDatasets/Preproccessed/' + str(curr_date.year) + '-' + str(curr_date.month) + '-' + str(curr_date.day) + '.csv'
        dataset = pd.read_csv(filename)
        curr_histogram = [0 for i in range(0, length)]
        for i in range(0, len(dataset)):
            num = dataset.at[i, colName]
            if num > values[length-1]:
                curr_histogram[length-1] = curr_histogram[length-1] + 1
                continue
            for j in range(0, length-1):
                if values[j] <= num <= values[j+1]:
                    curr_histogram[j] = curr_histogram[j] + 1
        if curr_date == start_date:
            histogram = curr_histogram
            curr_date = curr_date + timedelta(days=1)
            continue
        for j in range(0, length):
            histogram[j] = (histogram[j] + curr_histogram[j])/2
        curr_date = curr_date + timedelta(days=1)
    plt.bar(strings, histogram)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.15, bottom=0.35)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def verifiedExperiment():
    start_date = datetime.datetime(2018, 8, 17)
    curr_date = datetime.datetime(2018, 8, 17)
    end_data = datetime.datetime(2020, 12, 31)
    histogram = [0, 0]
    strings = ['Verified', 'Not Verified']
    while curr_date != end_data:
        filename = './tweetsDatasets/Preproccessed/' + str(curr_date.year) + '-' + str(curr_date.month) + '-' + str(curr_date.day) + '.csv'
        dataset = pd.read_csv(filename)
        curr_histogram = [0, 0]
        for i in range(0, len(dataset)):
            verified = dataset.at[i, 'VerifiedUser']
            if verified:
                curr_histogram[0] = curr_histogram[0] + 1
            else:
                curr_histogram[1] = curr_histogram[1] + 1
        if curr_date == start_date:
            histogram = curr_histogram
            curr_date = curr_date + timedelta(days=1)
            continue
        for j in range(0, 2):
            histogram[j] = (histogram[j] + curr_histogram[j])/2
        curr_date = curr_date + timedelta(days=1)
    plt.bar(strings, histogram)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.15, bottom=0.35)
    plt.title('Average Number of Verified Users')
    plt.ylabel('Number of Users')
    plt.xlabel('User is Verified or Not')
    plt.show()

def scoresExperiment():
    start_date = datetime.datetime(2018, 8, 17)
    curr_date = datetime.datetime(2018, 8, 17)
    end_data = datetime.datetime(2020, 12, 31)
    while curr_date != end_data:
        filename = './tweetsDatasets/Preproccessed/' + str(curr_date.year) + '-' + str(curr_date.month) + '-' + str(curr_date.day) + '.csv'
        dataset = pd.read_csv(filename)
        dataset.drop(dataset.loc[dataset['compound'] == 0].index, inplace=True)
        plt.hist(dataset['compound'], bins=200)
        plt.xlabel('Sentiment Analysis Score')
        plt.ylabel('Frequency')
        plt.show()
        curr_date = curr_date + timedelta(days=1)

if __name__ == '__main__':
    ylabel1 = 'Number of Users'
    followers = [0, 100, 1000, 5000, 10000, 30000, 50000, 100000, 1000000]
    col_name = 'UserFollowersCount'
    title1 = 'Average Number of Followers - Histogram'
    xlabel1 = 'Number of Followers'
    experiment(followers, col_name, title1, xlabel1, ylabel1)
    statuses = [0, 100, 1000, 5000, 10000, 30000, 50000, 100000, 300000, 1000000]
    col_name = 'UserStatusesCount'
    title1 = 'Average Number of Statuses - Histogram'
    xlabel1 = 'Number of Statuses'
    experiment(statuses, col_name, title1, xlabel1,ylabel1)
    friends = [0, 500, 1000, 2000, 3000, 5000]
    col_name = 'UserFriendsCount'
    title1 = 'Average Number of Friends - Histogram'
    xlabel1 = 'Number of Friends'
    experiment(friends, col_name, title1, xlabel1,ylabel1)
    favourites = [0, 100, 1000, 5000, 10000, 30000, 50000, 100000, 300000, 1000000]
    col_name = 'UserFavouritesCount'
    title1 = 'Average Number of User Favourites - Histogram'
    xlabel1 = 'Number of User\'s Favourites'
    experiment(favourites, col_name, title1, xlabel1,ylabel1)
    verifiedExperiment()
    scoresExperiment()


