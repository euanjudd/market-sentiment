#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parse processed tweet data for keywords.

@author: Euan Judd
@github: https://github.com/euanjudd/market-sentiment
"""

class TweetStatistics():
    """Parse tweets for keywords.

    Methods: keyword_counter(tweet, keyword)
    """

    def keyword_counter(self, tweet, keyword):
        """Counts the number of times a keyword appears in a tweet and returns an integer.

        Keyword arguments:
        tweet -- tweet data after all punctuation, stopwords, etc have been removed.
        keyword -- word to be counted
        """
        return tweet.count(keyword)