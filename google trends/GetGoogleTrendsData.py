from pytrends import dailydata

if __name__ == '__main__':
    # Getting google trends data about the search word 'ethereum' since August 2017
    # Until May 2021.
    # The data collected represents google searches in the United States
    df = dailydata.get_daily_data('ethereum', 2017, 8, 2021, 5, geo='US')
    df.to_csv('ethereumGoogleTrends.csv')















