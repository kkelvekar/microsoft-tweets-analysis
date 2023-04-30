import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# load the raw tweets
df = pd.read_csv('data/raw_tweets.csv')

# replace NaN values with an empty string
df = df.fillna('')

# vectorize the raw tweets
vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')
dtm = vectorizer.fit_transform(df['tweet'])

# create and fit the LDA model
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(dtm)

# create a dataframe with the topic distributions for each tweet
topic_distributions = pd.DataFrame(lda.transform(dtm))

# apply t-SNE to the topic distributions
tsne_model = TSNE(n_components=2, random_state=42)
tsne_data = tsne_model.fit_transform(topic_distributions)

# add a new column to the dataframe that labels each tweet with its most likely topic
df['Topic'] = topic_distributions.idxmax(axis=1)

# save the dataframe with the topic labels to a CSV file
df.to_csv('data/topic_modeling/tweets_with_topics.csv', index=False)

# print the top 15 words for each of the 10 topics and save to a CSV file
top_words_df = pd.DataFrame()
for i, topic in enumerate(lda.components_):
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-15:]]
    top_words_df[f'Topic_{i}'] = top_words

top_words_df.to_csv('data/topic_modeling/top_words.csv', index=False)

# create a scatter plot of the t-SNE results
plt.figure(figsize=(10, 10))
scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=df['Topic'], cmap='tab10')
plt.legend(*scatter.legend_elements(), title="Topics")
plt.savefig('figures/topic_modeling/tsne_plot.png')

# Aggregate tweets by Topic and product
df_grouped = df.groupby(['Topic', 'product'])['tweet'].apply(' '.join).reset_index()
df_grouped.to_csv('data/topic_modeling/tweets_with_topics_grouped.csv', index=False)

print("Topic Model Analysis complete. Data and visualizations saved in the data and figures directories.")
