#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Process raw tweets to remove everything except the most important words.

@author: Euan Judd
@github: https://github.com/euanjudd/market-sentiment
"""

import re                               # for matching regular expressions in strings
from string import punctuation          # for a list of punctuation data
import nltk                             # for processing natural language
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# # # # PROCESS TWEETS # # # #
class ProcessTweets():
    """Processes raw tweets by removes URLs, retweets, usernames, hashtags, emojis, repeated characters, punctuations, and stopwords ('a', 'an','the', etc.).

    Constructor argument: tweet
    Class variables: tweet; words_to_remove
    Methods: __str__(); replace_url(); remove_retweet(); replace_at_user(); replace_hashtag(); only_keep_ascii_characters(); remove_unwanted_words()
    """

    def __init__(self, tweet):
        """A new class for each tweet so that methods can be chained. Words to remove once tweet has been cleaned.

        Keyword arguments:
        tweet -- raw tweet data from Twitter.
        """
        self.tweet = tweet.lower()
        self.words_to_remove = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    def __str__(self):
        """Return processed tweet as a string"""
        return self.tweet

    def replace_url(self):
        """Replace URLs in a tweet with "URL" and return self."""
        self.tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', self.tweet)
        # MyCharacter(chr(self.value + other))
        return self

    def remove_retweet(self):
        """Removes retweets from tweet and return self."""
        self.tweet = re.sub('\s*rt\s', '', self.tweet)
        return self

    def replace_at_user(self):
        """Replace @user in tweet with AT_USER and return self."""
        self.tweet = re.sub('@[^\s]+', 'AT_USER', self.tweet)
        return self

    def replace_hashtag(self):
        """Replace #'s in tweet with \1 and return self."""
        self.tweet = re.sub(r'#([^\s]+)', r'\1', self.tweet)
        return self

    def only_keep_ascii_characters(self):
        """Encode a string to bytes, ignore what can't be converted, decode the bytes to a string and return self."""
        self.tweet = self.tweet.encode('ascii', 'ignore').decode('ascii')
        return self

    def remove_unwanted_words(self):
        """Keep words if they are not in the set of words to remove and return self."""
        self.tweet = word_tokenize(self.tweet)
        self.tweet = ' '.join([word for word in self.tweet if word not in self.words_to_remove])
        return self