import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import re
import numpy as np

# load the cleaned tweets
df = pd.read_csv('data/cleaned_tweets.csv')

# replace NaN values with an empty string
df = df.fillna('')

# create a wordcloud of the cleaned tweets
all_words = ' '.join([tweet for tweet in df['cleaned_tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.savefig('figures/exploratory_data_analysis/wordcloud.png')

# create a bar chart of the most frequently mentioned products
product_counts = Counter(df['product'])
product_df = pd.DataFrame.from_dict(product_counts, orient='index').reset_index()
product_df.columns = ['Product', 'Count']

product_df = product_df.sort_values('Count', ascending=False)
product_df.plot.bar(x='Product', y='Count')
plt.title('Most frequently mentioned products')
plt.xlabel('Product')
plt.ylabel('Count')
plt.savefig('figures/exploratory_data_analysis/product_mentions.png')

# create a table of the most frequently mentioned products
top_products = product_df.head(10)
print("Most frequently mentioned products:")
print(top_products)
top_products.to_csv('data/top_products.csv', index=False) # saving the top products to a csv file

# visualize trends in the tweets data
plt.figure(figsize=(12, 6))
sns.lineplot(data=df['product'].value_counts().sort_index())
plt.xticks(rotation=90)
plt.title("Trends in Tweets Data")
plt.xlabel("Product")
plt.ylabel("Count")
plt.savefig("figures/exploratory_data_analysis/tweets_trends.png")

def extract_hashtags(s):
    return re.findall(r'\#\w+', s)

# load the data
df = pd.read_csv('data/raw_tweets.csv')

# create a new column 'hashtags' that extracts hashtags from the cleaned tweets
df['hashtags'] = df['tweet'].apply(lambda x: extract_hashtags(x))

# create a list of all hashtags
all_hashtags = [hashtag for hashtags in df['hashtags'] for hashtag in hashtags]

# count and display the most common hashtags
counter = Counter(all_hashtags)
most_common_hashtags = counter.most_common(10)
print("Most common hashtags: ", most_common_hashtags)
most_common_hashtags_df = pd.DataFrame(most_common_hashtags, columns=['Hashtag', 'Count'])
most_common_hashtags_df.to_csv('data/most_common_hashtags.csv', index=False) # saving the most common hashtags to a csv file

# plot the most common hashtags
df_most_common_hashtags = pd.DataFrame(most_common_hashtags, columns=['hashtag', 'count'])
sns.barplot(x='count', y='hashtag', data=df_most_common_hashtags)

# plt.title('Top 10 hashtags')
# plt.xlabel('Count')
# plt.xticks(np.arange(0, max(df_most_common_hashtags['count']), step=10))  # Set x-ticks every 10 units
# plt.savefig('figures/exploratory_data_analysis/hashtags.png')

plt.title('Top 10 hashtags')
plt.xlabel('Count')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Add this line to remove x-axis values
plt.savefig('figures/exploratory_data_analysis/hashtags.png')

# Word cloud for most common hashtags
wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(dict(most_common_hashtags))

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('figures/exploratory_data_analysis/hashtags_wordcloud.png')

print("Exploratory Data Analysis complete. Visualizations and data saved in the figures and data directories.")


