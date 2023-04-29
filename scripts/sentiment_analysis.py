import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

def get_polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

def analyze_product_sentiment(product):
    product_tweets = df[df['product'].str.contains(product, case=False, na=False)].copy()  # Use copy() here
    product_tweets['Polarity'] = product_tweets['cleaned_tweet'].astype(str).apply(get_polarity)
    product_tweets['Sentiment'] = product_tweets['Polarity'].apply(get_sentiment)
    return product_tweets

def get_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df = pd.read_csv('data/cleaned_tweets.csv')
df = df.fillna('')

df['Polarity'] = df['cleaned_tweet'].astype(str).apply(get_polarity)
df['Sentiment'] = df['Polarity'].apply(get_sentiment)

# Save overall sentiment data to a CSV file
df.to_csv('data/sentiment_analysis/sentiment_analysis_overall.csv', index=False)

sns.countplot(x='Sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.savefig('figures/sentiment_analysis/sentiment_analysis_overall.png')

products = df['product'].unique()

for product in products:
    product_tweets = analyze_product_sentiment(product)
    # Save product sentiment data to a CSV file
    product_tweets.to_csv(f'data/sentiment_analysis/sentiment_analysis_{product}.csv', index=False)
    sns.countplot(x='Sentiment', data=product_tweets, order=['Positive', 'Neutral', 'Negative'])
    plt.title(f'Sentiment Analysis for {product}')
    plt.savefig(f'figures/sentiment_analysis/sentiment_analysis_{product}.png')
    plt.clf()

print("Sentiment Analysis complete. Data and visualizations saved in the data and figures directories.")
