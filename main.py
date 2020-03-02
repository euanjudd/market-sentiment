#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Correlates market sentiment from twitter with stock market price data.

@author: Euan Judd
@github: https://github.com/euanjudd/market-sentiment
"""
import pandas as pd                     # for working with datasets
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
    def __init__(self, tweet):
        """A new class for each tweet"""
        self.tweet = tweet.lower()
        self.words_to_remove = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def __str__(self):
        return self.tweet

    def replace_url(self):
        """Replace URLs in a tweet with "URL"."""
        self.tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', self.tweet)
        # MyCharacter(chr(self.value + other))
        return self

    def remove_retweet(self):
        """Removes retweets from tweet."""
        self.tweet = re.sub('\s*rt\s', '', self.tweet)
        return self

    def replace_at_user(self):
        """Replace @user in tweet with AT_USER."""
        self.tweet = re.sub('@[^\s]+', 'AT_USER', self.tweet)
        return self

    def replace_hashtag(self):
        """Replace #'s in tweet with \1."""
        self.tweet = re.sub(r'#([^\s]+)', r'\1', self.tweet)
        return self

    def only_keep_ascii_characters(self):
        """Encode a string to bytes, ignore what can't be converted, decode the bytes to a string and return."""
        self.tweet = self.tweet.encode('ascii', 'ignore').decode('ascii')
        return self

    def remove_unwanted_words(self):
        """Keep words if they are not in the set of words to remove."""
        self.tweet = word_tokenize(self.tweet)
        self.tweet = ' '.join([word for word in self.tweet if word not in self.words_to_remove])
        return self

# # # # TWEET STATISTICS # # # #
class TweetStatistics():
    """
    Analyse tweets for keyfords.
    """
    def keyword_counter(self, tweet, keyword):
        """Counts the number of times a keyword appears in a tweet."""
        return tweet.count(keyword)

# # # # FORWARD STEPWISE SELECTION # # # #
class FordwardStepwiseSelection():
    """
    Starts with zero predictors. A new predictor is added one-at-a-time. The best predictor is selected. A second predictor is
    then added one-at-a-time. The best pair is selected. etc.
    """
    def __init__(self, predictors):
        """A subset of predictors that haven't been tested this round."""
        self.subset = predictors.copy()

    def iterate_predictors(self, selection):
        """Returns the next predictor for testing."""
        self.subset = list(set(self.subset).difference(selection)) # Subset will be rearranged every time this is called
        idx = self.subset[0] # Make new selection
        self.subset.remove(idx) # Remove new selection
        return idx

    def reset_predictors(self, predictors):
        """Reset subset to the original set so the next round can start."""
        if len(self.subset) == 0:
            self.subset = predictors.copy()
            return True
        return False

# # # # STATISTICAL LEARNING # # # #
class StatisticalLearning():
    """
    Multiple statistical learning methods to find a relationship between multiple time series.
    """
    def __init__(self):
        pass

    def granger_causality(self):
        """Find the correlation and lag between two time series."""
        pass

    def recurrent_neural_network(self):
        """Find an approximation of the function that relates two time series."""
        pass

if __name__ == "__main__":
    """DATA"""
    gld_data_frame = pd.read_csv('GLD.csv') #'Data','Open','High','Low','Close','Adj Close','Volume'
    gld_dates = [dt.datetime.strptime(i, '%Y-%m-%d') for i in gld_data_frame['Date']]

    sentiment_data_frame = pd.read_csv('sentiment.csv') #'TweetDate','UserLocation','Tweets','Username','FavouriteCount',RTCount'
    sentiment_data_frame.sort_values(by=['TweetDate'], inplace=True, ascending=True)
    sentiment_dates = [dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in sentiment_data_frame['TweetDate']]

    """CLEAN TWEET DATA"""
    sentiment_data_frame['CleanedTweets'] = sentiment_data_frame['Tweets'].apply(lambda x:
        str(ProcessTweets(x).replace_url().remove_retweet().replace_at_user().replace_hashtag().only_keep_ascii_characters().remove_unwanted_words())
    )

    """PROCESS KEYWORD INFORMATION IN TWEETS"""
    keywords = ['Happy', 'Sad', 'Shitstorm', 'Bear', 'Bull']
    tweet_stats = TweetStatistics()
    for keyword in keywords: sentiment_data_frame[keyword] = sentiment_data_frame['CleanedTweets'].apply(lambda x: tweet_stats.keyword_counter(x, keyword.lower()))

    """FORWARD STEPWISE SELECTION"""
    fordward_stepwise_selection = FordwardStepwiseSelection(keywords)
    selection = []
    while len(selection) < len(keywords):
        test = fordward_stepwise_selection.iterate_predictors(selection)
        # Placeholder: Statistical learning method. Input:selection+test; Output:MSE.
        # Placeholder: Append [selection+test, MSE ].
        if fordward_stepwise_selection.reset_predictors(keywords):
            # Placeholder: Append [test, MSE] of best performing test to selection.
            selection.append(test)
    print(selection)

    """PLOT CLOSE PRICE AND RETWEETS ON THE SAME GRAPH"""
    # fig, ax1 = plt.subplots()
    # color = 'tab:red'
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel('Daily Close Price ($)', color=color)
    # ax1.plot(gld_dates, gld_data_frame['Close'], color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:blue'
    # ax2.set_ylabel('Happy', color=color)  # we already handled the x-label with ax1
    # ax2.plot(sentiment_dates, sentiment_data_frame[keywords], color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.legend(keywords)
    # plt.show()