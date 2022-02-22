# Ethereum-Prediction
The goal of our project is to try to predict the behavior of the ethereum, i.e. whether it will go up, down
or will not change on the next trading day. We would like to do so in three different and interesting ways:

1. In the first part, we will try to predict whether the ethereum is going up/down/not change in the next trading day, based on historical data.
We will do this with the help of statistical information on characteristics related to Ethereum and the blockchain network, periodic trading data and of course formulas
and well-known indicators that are an integral part of predictions in the world of stock market and crypto.

2. In the second part we would like to predict an increase / decrease / non-change in the Ethereum price according to textual information from Twitter,
That is, given the tweets that were posted on Twitter the day before or in the previous days, we will analyze the sentiment
of these tweets to understand with their help whether the etherium is in a positive or negative trend, in addition to the content
of these tweets and their sentiment, we will take into account very important characteristics that relate to the tweets themselves or
related to the user who posted the tweet, for example the number of likes on the tweet or the number of followers of the user who posted the tweet.

3. In third part, we will try to find a connection between the Google Trends values of a particular period and the behavior
Of the ethereum and we will try to predict, with the help of the information we receive from Google, the behavior of the ethereum for
This period (increase / decrease / no change). For each day, an increase / decrease / no change is predicted.
With the help of Google Trends from previous days and with the help of features that are based on this data.




What is the purpose of each file ? 
** We'll start with the folder that takes care of the tweets part which is "Tweets Code". **
1. CollectTweetsUsingOldTweets.py - the main purpose of this file to scrape tweets data from twitter since 16/08/2017 until 21/05/2021. 
for each one of these days we scrape up to 10,000 tweets including the tweet's content, likes, favorites and more features. 
Tweets relevant to a specific date are saved into one excel file. This means each day has a different excel file.

2. CreatingOneDatasetFromTweetsDatasets.py - the main purpose of this file is to take all data files that were created in the previous file and create one big dataset with many important features that represent
the data in the excel files of each day. 

3. TweetsDatasetsPreproccessing.py - this file is responsible for cleaning the data, cleaning the tweets from all irrelevant characters and links and any other things that might disturb the performance of our sentiment analyzer.
Such as : removing links, removing tags, removing stop words, correcting any spelling mistakes and more. 

4. TweetsExperiments.py - responsible for all experiments done on the data to visualize it and understand it, such as histograms of likes and verified users and more. 

5. tweetsFeatureSelection.py - responsible for feature selection, feature reduction using correlation experiments and model evaluation. 


** We'll continue with the next folder that takes care of the google trends part which is "Google Trends". **

1. GetGoogleTrendsData.py - the main purpose of this file to scrape google trends data data from google since 08/2017 until 05/2021.

2. googleTrendsDataProccessing.py - the main purpose of this file is to preproccess the data extracted from Google and create a dataset with the added features we chose based on the experiments we've done. The dataset created here will be used during our prediction using google trends. 

3. TrendsCorrelation.py - this file is responsible for the experiment that calculates the Correlation between each two features and draws a graph for each two features that have a high correlation and saves it. 

4. TrendsRandomForest.py - this file is responsible for the hyper-parameter tuning of the Random Forest model and for the  model evaluation for the google trends part.

** We'll continue with the next folder that takes care of the historical data part which is "Historical Code". **

1. HistoricalFilterByVariance.py - this file is responsible for the Variance experiment and finding out all features with low variance in order to get rid of them.

2. HistoricalPartCorrelation.py - responsible for the Correlation experiment between each two features. 

3. HistoricalFeatureSelection.py - responsible for selecting the best features for our historical data model using SFS.

4. HistoricalPartID3.py - the main purpose of this file is hyper-parameter tuning and model evalutaion of the ID3 Model.

5. HistoricalPartKNN.py - the main purpose of this file is hyper-parameter tuning and model evalutaion of the KNN Model.

6. HistoricalPartSVR.py - the main purpose of this file is hyper-parameter tuning and model evalutaion of the SVR Model.


Additional Files :

1. Google Trends/ Tweets Correlation experiments - is a zip code that includes all the graphs drawn for each two features with high correlation.
