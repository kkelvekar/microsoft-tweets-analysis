import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# load the cleaned tweets
df = pd.read_csv('data/cleaned_tweets.csv')

# replace NaN values with an empty string
df = df.fillna('')

# vectorize the cleaned tweets
vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')
dtm = vectorizer.fit_transform(df['cleaned_tweet'])

# create and fit the LDA model
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(dtm)

# print the top 15 words for each of the 10 topics and plot bar graphs
for i, topic in enumerate(lda.components_):
    print(f'Top 15 words for Topic #{i}:')
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-15:]]
    print(top_words)
    print('\n')

    # plot bar graph
    fig, ax = plt.subplots()
    ax.bar(range(15), topic.argsort()[-15:])
    ax.set_xticks(range(15))
    ax.set_xticklabels(top_words, rotation='vertical')
    plt.savefig(f'figures/topic_modeling/topic_{i}.png')

# add a new column to the dataframe that labels each tweet with its most likely topic
df['Topic'] = lda.transform(dtm).argmax(axis=1)

# save the dataframe with the topic labels to a CSV file
df.to_csv('data/topic_modeling/tweets_with_topics.csv', index=False)

# print log-likelihood and perplexity
print("Log likelihood: ", lda.score(dtm))
print("Perplexity: ", lda.perplexity(dtm))

print("Topic Model Analysis complete. Data and visualizations saved in the data and figures directories.")
