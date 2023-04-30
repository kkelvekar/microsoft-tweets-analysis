import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

def get_polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

def analyze_product_sentiment(product):
    product_tweets = df[df['product'].str.contains(product, case=False, na=False)].copy()
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

products = ['Overall', 'Edge', 'Teams', 'Outlook', 'OneDrive', 'Windows', 'GitHub', 'Xbox', '.NET', 'Azure', 'PowerBI']

fig, axes = plt.subplots(4, 3, figsize=(15,20))
axes = axes.ravel()

product_sentiments = []

for i, product in enumerate(products):
    if product == 'Overall':
        product_tweets = df.copy()
    else:
        product_tweets = analyze_product_sentiment(product)

    product_tweets['product'] = product
    product_sentiments.append(product_tweets)
    sns.countplot(x='Sentiment', data=product_tweets, order=['Positive', 'Neutral', 'Negative'], ax=axes[i])
    if product == 'Overall':
        axes[i].set_title(f'Sentiment Analysis for Microsoft (Overall)')
    else:
        axes[i].set_title(f'Sentiment Analysis for Microsoft {product}')
    
# Remove the last two unused subplots
fig.delaxes(axes[10])
fig.delaxes(axes[11])

plt.tight_layout()
plt.savefig('figures/sentiment_analysis/sentiment_analysis_combined.png')

df_combined = pd.concat(product_sentiments, ignore_index=True)
sentiment_counts = df_combined.groupby(['product', 'Sentiment']).size().reset_index(name='count')

sentiment_counts.to_csv('data/sentiment_analysis/sentiment_analysis_combined.csv', index=False)

print("Sentiment Analysis complete. Data and visualizations saved in the data and figures directories.")
