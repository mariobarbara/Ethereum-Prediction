import os
import pandas as pd
import GetOldTweets3 as got
import snscrape.modules.twitter as sntwitter
import datetime
from datetime import timedelta
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from autocorrect import Speller
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def cleanDataset(dataset):
    stemmer = PorterStemmer()
    spell = Speller(lang='en')
    rows = len(dataset)
    for i in range(0, rows):
        text = dataset.at[i, 'Text']
        text = re.sub("#[A-Za-z0-9_]+", "", text)
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
        text = re.sub(r'@([A-Za-z0-9_]+)', '', text, flags=re.MULTILINE)
        text = re.sub('[^A-Za-z!?$\U0001F680\U0001F4C8\U0001F4C9]', ' ', text)
        text = text.lower()
        tokenized_tweet = word_tokenize(text)
        for word in tokenized_tweet:
            if word in stopwords.words('english'):
                tokenized_tweet.remove(word)
        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = stemmer.stem(tokenized_tweet[i])
            tokenized_tweet[i] = stemmer.stem(spell(tokenized_tweet[i]))
        result_tweet = " ".join(tokenized_tweet)
        text = result_tweet
        dataset.at[i, 'newText'] = text
    sid = SentimentIntensityAnalyzer()
    dataset['scores'] = dataset['newText'].apply(lambda tweet: sid.polarity_scores(tweet))
    dataset.head()
    dataset['compound'] = dataset['scores'].apply(lambda score_dict: score_dict['compound'])
    dataset.head()
    nan_value = float("NaN")
    dataset.replace("", nan_value, inplace=True)
    dataset.dropna(subset=["newText"], inplace=True)
    return dataset


if __name__ == '__main__':
    curr_date = datetime.datetime(2018, 10, 11)
    end_data = datetime.datetime(2020, 12, 12)
    while curr_date != end_data:
        curr_date_str = './tweetsDatasets/' + str(curr_date.year) + '-' + str(curr_date.month) + '-' + str(curr_date.day) + '.csv'
        dataset = pd.read_csv(curr_date_str)
        dataset['newText'] = ""
        newDataset = cleanDataset(dataset)
        filename = './tweetsDatasets/Preproccessed/' + str(curr_date.year) + '-' + str(curr_date.month) + '-' + str(curr_date.day) + '.csv'
        curr_date = curr_date + timedelta(days=1)
        newDataset.to_csv(filename)
