import pandas as pd

# load the combined tweets
df = pd.read_csv('data/tweets.csv')

# create a dataframe for raw tweets
raw_tweets = df[['tweet', 'product', 'reply']]
raw_tweets.to_csv('data/raw_tweets.csv', index=False)

# create a dataframe for cleaned tweets
cleaned_tweets = df[['cleaned_tweet', 'product', 'reply']]
cleaned_tweets.to_csv('data/cleaned_tweets.csv', index=False)
