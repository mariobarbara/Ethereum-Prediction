import pandas as pd
import datetime

def getRealClass(open_price, close_price):
    if close_price >= open_price:
        return 1
    else:
        return -1

def getRealClassWithMargins(open_price, close_price):
    if open_price * 0.99 <= close_price <= 1.01 * open_price:
        return 0
    if close_price > open_price:
        return 1
    if close_price <= open_price:
        return -1

# Gets the path of the file of date. can open files of previous days by using the parameter prev_days.
# for example, prev_days = 1 will return the path of the day before the given date
def getRelevantFilePath(date, prev_days):
    day, month, year = date[0:2], date[3:5], date[6:10]
    curr_date = datetime.datetime(int(year), int(month), int(day))
    prev_date = curr_date - datetime.timedelta(days=prev_days)
    prev_day, prev_month, prev_year = str(prev_date.day), str(prev_date.month), str(prev_date.year)
    file_path = './tweetsDatasets/Preproccessed/' + prev_year + '-' + prev_month + '-' + prev_day + '.csv'
    return file_path

def getValues(tweets, ranges, category, column):
    rows = len(tweets)
    compounds = [-1, -0.1, 0, 0.1, 1]
    col_names = []
    values = []
    for i in range(0, len(ranges)-1):
        low, high = ranges[i], ranges[i+1]
        for j in range(0, len(compounds)-1):
            low_comp, high_comp = compounds[j], compounds[j+1]
            col = category + '[' + str(low) + ',' + str(high) + ']' + 'Compound[' + str(low_comp) + ',' + str(high_comp) + ')'
            result = tweets[(low_comp <= tweets['compound']) & (tweets['compound'] < high_comp) & (low <= tweets[column]) & (tweets[column] < high)]
            val = len(result)/rows
            col_names.append(col)
            values.append(val)
    for j in range(0, len(compounds) - 1):
        low_comp, high_comp = compounds[j], compounds[j + 1]
        result = tweets.loc[(ranges[len(ranges)-1] <= tweets[column]) & (low_comp <= tweets['compound']) & (tweets['compound'] < high_comp)]
        col = category + 'Over' + str(ranges[len(ranges)-1]) + 'Compound' + '[' + str(low_comp) + ',' + str(high_comp) + ')'
        val = len(result)/rows
        col_names.append(col)
        values.append(val)
    return col_names, values


if __name__ == '__main__':
    dataset = pd.read_csv("Binance_ethereum.csv")
    added_columns = ['label', 'label1']
    user_features = ['UserFollowersCount', 'UserFriendsCount', 'UserFavouritesCount', 'UserStatusesCount']
    max_features = ['UserFollowersCount', 'RetweetCount', 'LikeCount']
    for col in added_columns:
        dataset[col] = ""
    followers_ranges = [0, 1000, 5000, 30000]
    statuses_ranges = [0, 5000, 30000]
    likes_ranges = [0, 1000, 5000, 10000, 20000, 50000]

    for i in range(0, len(dataset)):
        date, open, close = dataset.at[i, 'date'], dataset.at[i, 'open'], dataset.at[i, 'close']
        dataset.at[i, 'label'] = getRealClass(open, close)
        dataset.at[i, 'label1'] = getRealClassWithMargins(open, close)
        filepath = getRelevantFilePath(date, 1)
        tweets_dataset = pd.read_csv(filepath)
        tweets_dataset = tweets_dataset.loc[tweets_dataset['compound'] != 0]
        tweets_dataset['UserWeight'] = tweets_dataset['VerifiedUser'] * tweets_dataset['compound']
        for feature in user_features:
            colname = 'Average' + feature
            dataset.at[i, colname] = sum(tweets_dataset[feature])/len(tweets_dataset[feature])
        dataset.at[i, 'PercentageOfVerifiedUsers'] = tweets_dataset['VerifiedUser'].value_counts(normalize=True)[1]
        for feature in max_features:
            colname = 'max' + feature + 'Score'
            idx = tweets_dataset[feature].idxmax()
            score = tweets_dataset.at[idx, 'compound']
            dataset.at[i, colname] = score
        dataset.at[i, 'AvgUserWeight'] = sum(tweets_dataset['UserWeight'])/len(tweets_dataset['UserWeight'])
        dataset.at[i, 'AvgReplyCount'] = sum(tweets_dataset['ReplyCount'])/len(tweets_dataset['ReplyCount'])
        x = getValues(tweets_dataset, followers_ranges, 'Followers', 'UserFollowersCount')
        for j in range(0, len(x[0])):
            dataset.at[i, x[0][j]] = x[1][j]
        y = getValues(tweets_dataset, statuses_ranges, 'UserStatuses', 'UserStatusesCount')
        for j in range(0, len(y[0])):
            dataset.at[i, y[0][j]] = y[1][j]
        z = getValues(tweets_dataset, likes_ranges, 'Likes', 'LikeCount')
        for j in range(0, len(z[0])):
            dataset.at[i, z[0][j]] = z[1][j]
    dataset.to_csv("dataset_tweets_part.csv")

