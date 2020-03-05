#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Correlates market sentiment from twitter with stock market price data.

@author: Euan Judd
@github: https://github.com/euanjudd/market-sentiment
"""

import pandas as pd                     # for working with datasets
import datetime as dt                   # for setting dates string to datetime.datetime object
from process_tweets import ProcessTweets
from tweet_statistics import *
from forward_stepwise_selection import *
from statistical_learning import *
from multi_line_plot import *

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
    forward_stepwise_selection = ForwardStepwiseSelection(keywords)
    selection = []
    while len(selection) < len(keywords):
        predictor = forward_stepwise_selection.iterate_predictors(selection)
        # Placeholder: Statistical learning method. Input:selection+predictor; Output:MSE.
        # Placeholder: Append [selection+predictor, MSE ].
        if forward_stepwise_selection.finished_cycle(keywords):
            # Placeholder: Append [predictor, MSE] of best performing test to selection.
            selection.append(predictor)
    print(selection)

    """PLOT CLOSE PRICE AND RETWEETS ON THE SAME GRAPH"""
    MultiLinePlot().plot_dataframes(x1 = gld_dates, y1 = gld_data_frame['Close'], y1_label = 'Daily Close Price ($)',
                    x2 = sentiment_dates, y2 = sentiment_data_frame[keywords], y2_label = 'Happy')