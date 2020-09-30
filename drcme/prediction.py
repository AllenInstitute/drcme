"""
The :mod:`drcme.prediction` module contains wrapper functions for random
forest classification.
"""

import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import logging


def rf_predict(train_df, train_labels, test_df, n_trees=500, class_weight=None):
    """Predict labels for `test_df` by random forest classification

    Trains a classifier using `train_df` and `train_labels`, then predicts
    labels for `test_df`.

    Parameters
    ----------
    train_df : DataFrame
        Training data
    train_labels : list or array
        Labels for training data
    test_df : DataFrame
        Test data
    n_trees : int, optional
        Number of trees for random forest classifier
    class_weight : {“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
        Class weight parameter for random forest classifier

    Returns
    -------
    array
        Predicted labels for `test_df`
    """
    rf = ensemble.RandomForestClassifier(n_estimators=n_trees, oob_score=True,
                                         class_weight=class_weight, random_state=0)
    rf.fit(train_df.values, train_labels)
    logging.info("OOB score: {:f}".format(rf.oob_score_))
    return rf.predict(test_df.values)


