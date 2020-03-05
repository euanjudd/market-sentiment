#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot multiple pandas dataframe time series on a single plot.

@author: Euan Judd
@github: https://github.com/euanjudd/market-sentiment
"""

import matplotlib.pyplot as plt         # for plotting

class MultiLinePlot():
    """Plot two time series on a single plot.

    Constructor argument: x1, y1, x2, y2 Pandas dataframes.
    Class variables: same
    Methods: plot()
    """

    def plot_dataframes(self, x1, y1, y1_label, x2, y2, y2_label):
        """x1, y1, x2, y2 Pandas dataframes."""
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(y1_label, color=color)
        ax1.plot(x1, y1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel(y2_label, color=color)  # we already handled the x-label with ax1
        ax2.plot(x2, y2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.legend(keywords)
        plt.show()