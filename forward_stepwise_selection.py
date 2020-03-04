#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Forward stepwise selection of machine learning predictors.

@author: Euan Judd
@github: https://github.com/euanjudd/market-sentiment
"""

class ForwardStepwiseSelection():
    """Forward stepwise selection starts with zero items and adds items one-at-a-time to find the best combination.

    Starts with zero predictors. A new predictor is added one-at-a-time. The best new predictor is selected. A second predictor is
    then added one-at-a-time. The best second predictor is selected, and so on.

    Constructor argument: predictors
    Class variables: sebset
    Methods: iterate_predictors(selection); finished_cycle(predictors)
    """

    def __init__(self, predictors):
        """Subset of predictors that haven't been tested this cycle."""
        self.subset = predictors.copy()

    def iterate_predictors(self, selection):
        """Returns the next predictor for testing.

        Keyword arguments:
        selection -- list of keywords useful as predictors
        """
        self.subset = list(set(self.subset).difference(selection)) # Subset will be rearranged every time this is called
        idx = self.subset[0] # Make new selection
        self.subset.remove(idx) # Remove new selection
        return idx

    def finished_cycle(self, predictors):
        """Ends cycle by resetting subset to the original set of predictors before the next cycle starts.

        Keyword arguments:
        predictors -- list of all keywords
        """
        if len(self.subset) == 0:
            self.subset = predictors.copy()
            return True
        return False
