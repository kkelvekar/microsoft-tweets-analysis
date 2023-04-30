import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_numeric, remove_stopwords, strip_short, stem_text
import matplotlib.pyplot as plt
import os

# load the raw tweets
df = pd.read_csv('data/raw_tweets.csv')

# replace NaN values with an empty string
df = df.fillna('')

# define the list of filters to apply
FILTERS = [strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short, stem_text]

# apply the filters to the tweet column
df['cleaned_tweet'] = df['tweet'].apply(lambda x: preprocess_string(x, FILTERS))

# create a dictionary and corpus for the LDA model
id2word = corpora.Dictionary(df['cleaned_tweet'])
corpus = [id2word.doc2bow(text) for text in df['cleaned_tweet']]

# create and fit the LDA model
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10, random_state=42, passes=10)

# save the dataframe with the topic labels to a CSV file
df['Topic'] = [sorted(lda_model.get_document_topics(doc), key=lambda x: -x[1])[0][0] for doc in corpus]
df.to_csv('data/topic_modeling/tweets_with_topics.csv', index=False)

# create a bar chart for each of the top 9 topics
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(5.8, 5.8))  # Adjusted to half the A4 size

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']

for i, ax in enumerate(axes.flatten()):
    terms = lda_model.get_topic_terms(i, 10)
    words = [id2word[term[0]] for term in terms]
    weights = [term[1] for term in terms]

    ax.barh(words, weights, color=colors[i])
    ax.invert_yaxis()
    ax.set_title(f'Top 10 Words for Topic {i}', fontsize=7)
    ax.set_xlabel('Weight', fontsize=4.5)
    ax.set_ylabel('Word', fontsize=4.5)
    ax.tick_params(labelsize=4)

plt.tight_layout()
plt.savefig('figures/topic_modeling/topics_grid.png', dpi=300)  # Reduced DPI to 300

# print the top 10 words for each of the 9 topics
df_topics_list = []
for i in range(9):
    terms = lda_model.get_topic_terms(i, 10)
    # add the topic and its terms to the list
    df_topics_list.append(pd.DataFrame({'Topic': [i], 'Top Terms': [[id2word[term[0]] for term in terms]]}))

# concatenate all the dataframes in the list
df_topics = pd.concat(df_topics_list, ignore_index=True)

# write the DataFrame to a CSV file
df_topics.to_csv('data/topic_modeling/topic_terms.csv', index=False)

# print the top 10 words for each of the 9 topics
df_topics_list = []
for i in range(9):
    terms = lda_model.get_topic_terms(i, 10)
    words = [id2word[term[0]] for term in terms]
    weights = [term[1] for term in terms]
    # add the topic, terms and their weights to the list
    df_topics_list.append(pd.DataFrame({'Topic': [i]*10, 'Term': words, 'Weight': weights}))

# concatenate all the dataframes in the list
df_topics = pd.concat(df_topics_list, ignore_index=True)

# write the DataFrame to a CSV file
df_topics.to_csv('data/topic_modeling/topic_terms_weights.csv', index=False)

print("Topic Model Analysis complete. Data and visualizations saved in the data and figures directories.")
