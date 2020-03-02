#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Correlates market sentiment from twitter with stock market price data.

@author: Euan Judd
@github: https://github.com/euanjudd/market-sentiment
"""
import pandas as pd                     # for working with datasets
from textblob import TextBlob           # for analysing sentiment of text data
import matplotlib.pyplot as plt         # for plotting
import datetime as dt                   # for setting dates string to datetime.datetime object
import random                           # for generating random numbers
import re                               # for matching regular expressions in strings
from string import punctuation          # for a list of punctuation data
import nltk                             # for processing natural language
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# # # # PROCESS TWEETS # # # #
class ProcessTweets():
    """
    Process raw tweets by removes URLs, retweets, usernames, hashtags, emojis, repeated characters, punctuations, and
    stopwords ('a', 'an','the', etc.)
    """
    def replace_url(self, tweet):
        """Replace URLs in a tweet with "URL"."""
        return re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)

    def remove_retweet(self, tweet):
        """Removes retweets from tweet."""
        return re.sub('\s*rt\s', '', tweet)

    def replace_at_user(self, tweet):
        """Replace @user in tweet with AT_USER."""
        return re.sub('@[^\s]+', 'AT_USER', tweet)

    def replace_hashtag(self, tweet):
        """Replace #'s in tweet with \1."""
        return re.sub(r'#([^\s]+)', r'\1', tweet)

    def only_keep_ascii_characters(self, tweet):
        """Encode a string to bytes, ignore what can't be converted, decode the bytes to a string and return."""
        return tweet.encode('ascii', 'ignore').decode('ascii')

    def char_to_remove(self):
        """Form a set of characters we want to remove from a tweet."""
        return set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def remove_unwanted_words(self, tweet):
        """Keep words if they are not in the set of words to remove."""
        tweet = word_tokenize(tweet)
        words_to_remove = self.char_to_remove()
        return ' '.join([word for word in tweet if word not in words_to_remove])

# # # # SENTIMENT ANALYSIS # # # #
class SentimentAnalysis():
    """
    Sentiment analysis of string using keywords
    """
    def __init__(self):
        """Keywords of interest."""
        self.keywords = ["happy", "sad", "shitstorm", "beating", "bull", "bear"]

    def sentiment(self, tweet):
        """Determines if the tweet is bullish or bearish"""
        pass

if __name__ == "__main__":
    # PRICE DATA #
    gld_data_frame = pd.read_csv('GLD.csv') #'Data','Open','High','Low','Close','Adj Close','Volume'
    # CLEAN TWEET DATA #
    sentiment_data_frame = pd.read_csv('sentiment.csv') #'TweetDate','UserLocation','Tweets','Username','FavouriteCount',RTCount'
    sentiment_data_frame.sort_values(by=['TweetDate'], inplace=True, ascending=True)
    process_tweets = ProcessTweets()
    sentiment_data_frame['Tweets'] = sentiment_data_frame['Tweets'].apply(lambda x: x.lower())
    sentiment_data_frame['CleanedTweets'] = sentiment_data_frame['Tweets'].apply(lambda x: process_tweets.replace_url(x))
    sentiment_data_frame['CleanedTweets'] = sentiment_data_frame['CleanedTweets'].apply(lambda x: process_tweets.remove_retweet(x))
    sentiment_data_frame['CleanedTweets'] = sentiment_data_frame['CleanedTweets'].apply(lambda x: process_tweets.replace_at_user(x))
    sentiment_data_frame['CleanedTweets'] = sentiment_data_frame['CleanedTweets'].apply(lambda x: process_tweets.replace_hashtag(x))
    sentiment_data_frame['CleanedTweets'] = sentiment_data_frame['CleanedTweets'].apply(lambda x: process_tweets.only_keep_ascii_characters(x))
    sentiment_data_frame['CleanedTweets'] = sentiment_data_frame['CleanedTweets'].apply(lambda x: process_tweets.remove_retweet(x))
    sentiment_data_frame['CleanedTweets'] = sentiment_data_frame['CleanedTweets'].apply(lambda x: process_tweets.remove_unwanted_words(x))

    # ANALYSE CLEANED TWEETS #
    sentiment_analysis = SentimentAnalysis()
    # sentiment_data_frame['TweetPolarity'] = sentiment_data_frame['CleanedTweet'].apply(lambda x: TextBlob(x).sentiment.polarity)  # Sentiment polarity [-1,1]
    sentiment_data_frame['TweetSentiment'] = sentiment_data_frame['CleanedTweets'].apply(lambda x: sentiment_analysis.sentiment(x))  # Sentiment [-1,1]

    gld_dates = [dt.datetime.strptime(i, '%Y-%m-%d') for i in gld_data_frame['Date']]
    sentiment_dates = [dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in sentiment_data_frame['TweetDate']]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Close Price ($)', color=color)
    ax1.plot(gld_dates, gld_data_frame['Close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Retweets', color=color)  # we already handled the x-label with ax1
    ax2.plot(sentiment_dates, sentiment_data_frame['RTCount'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()