import numpy as np
import pandas as pd
import sklearn.ensemble as ensemble
import logging

def rf_predict(train_df, train_labels, test_df, n_trees=500, class_weight=None):
    rf = ensemble.RandomForestClassifier(n_estimators=n_trees, oob_score=True,
                                         class_weight=class_weight, random_state=0)
    rf.fit(train_df.values, train_labels)
    logging.info("OOB score: {:f}".format(rf.oob_score_))
    return rf.predict(test_df.values)


