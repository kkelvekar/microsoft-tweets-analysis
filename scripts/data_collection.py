import time
import tweepy
import pandas as pd

# setup tweepy with your Twitter API credentials
auth = tweepy.OAuthHandler("k760ICyneP5YxEqW7ZVMLQohG", "hn1TipZcSuAanO1glOlsZR218Hc2Dk1F7aUvTwRAN5FxMJsybx")
auth.set_access_token("119106674-4wdi8J44oRJ8vO7SvBYyiz0w5qO356QiOf7GAnPb", "Ychqr6tNAyp2Mp42M5TUM7COeSTD8w1O6K9iv5m8oaVDA")

api = tweepy.API(auth)

# define the search terms
search_terms = [
    'Microsoft', 'Azure', 'Office365', 'Windows', 'Surface',
    'Xbox', 'Outlook', 'SharePoint', 'Teams', 'Dynamics365',
    'PowerBI', 'VisualStudio', 'GitHub', 'OneDrive', 'Bing',
    'Edge', 'Skype', 'SQLServer', 'WindowsServer',
    'Microsoft365', '.NET', 'PowerShell', 'InternetExplorer'
]

# create an empty list
data = []

# loop through search terms
for term in search_terms:
    try:
        tweets = tweepy.Cursor(api.search_tweets, q=term, lang='en', tweet_mode='extended').items(200)
        for tweet in tweets:
            if 'retweeted_status' in tweet._json:
                tweet_text = tweet.retweeted_status.full_text
            else:
                tweet_text = tweet.full_text
            data.append({'tweet': tweet_text, 'product': term})

            # get the top 10 replies
            replies = tweepy.Cursor(api.search_tweets, q='to:' + tweet.user.screen_name,
                                    since_id=tweet.id, tweet_mode='extended').items(10)
            for reply in replies:
                if not hasattr(reply, 'in_reply_to_status_id_str'):
                    continue
                if reply.in_reply_to_status_id == tweet.id:
                    data.append({'tweet': reply.full_text, 'product': term, 'reply': 'reply'})

    except tweepy.RateLimitError:
        print("Rate limit reached. Saving collected data to CSV and sleeping for 15 minutes.")
        df = pd.DataFrame(data)
        df.to_csv('raw_tweets.csv', index=False)
        time.sleep(15 * 60)  # Sleep for 15 minutes
    except Exception as e:
        print(f"An error occurred: {e}")

# convert the list to a dataframe and save it to a CSV file
df = pd.DataFrame(data)
df.to_csv('raw_tweets.csv', index=False)
