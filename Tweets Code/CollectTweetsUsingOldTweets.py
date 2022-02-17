import os
import pandas as pd
import GetOldTweets3 as got
import snscrape.modules.twitter as sntwitter
import datetime
from datetime import timedelta

if __name__ == '__main__':

    curr_date = datetime.datetime(2017, 8, 16)
    # end_data = datetime.datetime(2017, 8, 21)

    end_data = datetime.datetime(2017, 8, 17)
    while curr_date != end_data:
        tweets_list2 = []
        curr_date_str = str(curr_date.year) + '-' + str(curr_date.month) + '-' + str(curr_date.day)
        next_date = curr_date + timedelta(days=1)
        next_date_str = str(next_date.year) + '-' + str(next_date.month) + '-' + str(next_date.day)
        query = 'ethereum since:' + curr_date_str + ' until:' + next_date_str
        # Using TwitterSearchScraper to scrape data and append tweets to list
        for i, tweet in enumerate(
                sntwitter.TwitterSearchScraper(query).get_items()):
            if i > 10000:
                break
            # if tweet.likeCount < 100:
            #     limit = limit + 1
            #     continue
            tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.user.verified,
                                 tweet.user.followersCount, tweet.user.friendsCount, tweet.user.favouritesCount,
                                 tweet.user.statusesCount, tweet.replyCount, tweet.retweetCount, tweet.likeCount])

        # Creating a dataframe from the tweets list above
        tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'VerifiedUser',
                                                         'UserFollowersCount', 'UserFriendsCount', 'UserFavouritesCount',
                                                         'UserStatusesCount', 'ReplyCount', 'RetweetCount', 'LikeCount'])
        filename = './tweetsDatasets/' + curr_date_str + '.csv'
        tweets_df2.to_csv(filename)
        curr_date = curr_date + timedelta(days=1)