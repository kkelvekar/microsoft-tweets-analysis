import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

# define the stop words
stop_words = set(stopwords.words('english'))

def clean_tweet(tweet):
    # lowercasing
    tweet = tweet.lower()
    # remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # remove @mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # remove special characters and numbers
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\w*\d\w*', '', tweet)
    # remove small words
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # tokenize the tweet
    tweet_tokens = word_tokenize(tweet)
    # remove stop words and lemmatize
    cleaned_tweet = [lemmatizer.lemmatize(word) for word in tweet_tokens if word not in stop_words]
    return " ".join(cleaned_tweet)

# load the raw tweets
df = pd.read_csv('raw_tweets.csv')

# clean the tweets
df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)

# save the cleaned tweets to a new CSV file
df.to_csv('cleaned_tweets.csv', index=False)
