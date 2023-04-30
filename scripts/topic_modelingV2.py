import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_numeric, remove_stopwords, strip_short, stem_text
import pyLDAvis.gensim_models
import os
import time
from selenium import webdriver

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

# visualize the topics
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'figures/topic_modeling/topics.html')

# create a new instance of google chrome
driver = webdriver.Chrome()  # Add path to your chromedriver if required

# set the window size
driver.set_window_size(1920, 1080)

# open the webpage
driver.get('file://' + os.path.realpath('figures/topic_modeling/topics.html'))

# give it some time to load all scripts
time.sleep(5)

# take screenshot
driver.save_screenshot('figures/topic_modeling/topics.png')

# close the chrome instance
driver.quit()

# print the top 15 words for each of the 10 topics
for i in range(lda_model.num_topics):
    terms = lda_model.get_topic_terms(i, 15)
    # print(f"Topic {i} Top Terms:", [id2word[term[0]] for term in terms])

# save the dataframe with the topic labels to a CSV file
df['Topic'] = [sorted(lda_model.get_document_topics(doc), key=lambda x: -x[1])[0][0] for doc in corpus]
df.to_csv('data/topic_modeling/tweets_with_topics.csv', index=False)

# print the top 15 words for each of the 10 topics
df_topics_list = []
for i in range(lda_model.num_topics):
    terms = lda_model.get_topic_terms(i, 15)
    # add the topic and its terms to the list
    df_topics_list.append(pd.DataFrame({'Topic': [i], 'Top Terms': [[id2word[term[0]] for term in terms]]}))

# concatenate all the dataframes in the list
df_topics = pd.concat(df_topics_list, ignore_index=True)

# write the DataFrame to a CSV file
df_topics.to_csv('data/topic_modeling/topic_terms.csv', index=False)

print("Topic Model Analysis complete. Data and visualizations saved in the data and figures directories.")
