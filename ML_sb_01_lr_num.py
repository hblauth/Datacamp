# Simple first model
#   Numeric columns only
#   Multi-class logistic regression model
#       Treats each label column as independent
#       Trains separate classifier on each label

# Due to nature of this dataset, usual train-test split won’t work as some labels only appear in a small fraction of the training set
#   If feature omitted from training set, can’t be predicted
#   Solution is to use StratifiedShuffleSplit
#       But only works with a single target variable
#       Create own function: multilevel_train_test_split()

# Multilevel_train_test_split function #

import numpy as np
import pandas as pd

def multilabel_sample(y, size=1000, min_count=5, seed=None):
""" Takes a matrix of binary labels `y` and returns the indices for a sample of size `size` if `size` > 1 or `size` * len(y) if size =< 1.
The sample is guaranteed to have > `min_count` of each label."""

    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).all():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('multilabel_sample only works with binary indicator matrices')

    if (y.sum(axis=0) < min_count).any():
        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')

    if size <= 1:
        size = np.floor(y.shape[0] * size)

    if y.shape[1] * min_count > size:
        msg = "Size less than number of columns * min_count, returning {} items instead of {}."
        warn(msg.format(y.shape[1] * min_count, size))
        size = y.shape[1] * min_count

        rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))

    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(y.shape[0])

        sample_idxs = np.array([], dtype=choices.dtype)

#first, guarantee > min_count of each label#

    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])

    sample_idxs = np.unique(sample_idxs)

#now that we have at least min_count of each, we can just random sample#

    sample_count = int(size - sample_idxs.shape[0])

#get sample_count indices from remaining choices#

    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices, size=sample_count,replace=False)


    return np.concatenate([sample_idxs, remaining_sampled]

def multilabel_sample_dataframe(df, labels, size, min_count=5, seed=None):
""" Takes a dataframe `df` and returns a sample of size `size` where all classes in the binary matrix `labels` are represented at least `min_count` times."""

    idxs = multilabel_sample(labels, size=size, min_count=min_count, seed=seed)

    return df.loc[idxs

def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):
""" Takes a features matrix `X` and a label matrix `Y` and returns (X_train, X_test, Y_train, Y_test) where all
classes in Y are represented at least `min_count` times. """
    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])

    test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
    train_set_idxs = np.setdiff1d(index, test_set_idxs)

    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask

    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])

# Training the simple model

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

#Create the DataFrame
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

#Get labels and convert to dummy variables
label_dummies = pd.get_dummies(df[LABELS])

#Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,label_dummies, size=0.2,seed=123)

#Instantiate the classifier
clf = OneVsRestClassifier(LogisticRegression())

#Fit the classifier to the training data
clf.fit(X_train, y_train)

# For accuracy:
# print("Accuracy: {}".format(clf.score(X_test, y_test)))

# We want the probability of the prediction, not just 1 or 0

#Load holdout data
holdout = pd.read_csv('HoldoutData.csv', index_col=0)

#Generate predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

#Format predictions in DataFrame
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns, index=holdout.index, data=predictions)

#Save to csv
prediction_df.to_csv('predictions.csv')

#Submit predictions for scoring
score = score_submission(pred_path='predictions.csv')

print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))
