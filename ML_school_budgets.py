## Encode labels as categorical variables ##
'''Define the lambda function'''
categorize_label = lambda x: x.astype('category')

'''Convert df[LABELS] to a categorical type'''
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)

# Perform EDA to understand distribution of unique labels
import matplotlib.pyplot as plt

'''Calculate number of unique values for each label'''
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

'''Plot number of unique values for each label'''
num_unique_labels.plot(kind='bar')
plt.xlabel('Labels')
plt.ylabel('Number of unique values')
plt.show()

## Creating a simple first model

    #* Train basic model on numeric columns only
    #* Here, use a multi-class logistic regression model
    #    * Treats each label column as independent
    #    * Trains separate classifier on each label
    #* Due to nature of this dataset, usual train-test split won’t work as some labels only appear in a small fraction of the training set
    #    * If omitted from training set, then can’t be predicted
    #    * Solution is to use StratifiedShuffleSplit
    #        * But only works with a single target variable
    #        * Create own function: multilevel_train_test_split()

##Create multilevel_train_test_split function ##

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

'''first, guarantee > min_count of each label'''

    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])

    sample_idxs = np.unique(sample_idxs)

'''now that we have at least min_count of each, we can just random sample'''

    sample_count = int(size - sample_idxs.shape[0])

'''get sample_count indices from remaining choices'''

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

## Training the simple model ##

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

'''Create the DataFrame'''
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

'''Get labels and convert to dummy variables'''
label_dummies = pd.get_dummies(df[LABELS])

'''Create training and test sets'''
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,label_dummies, size=0.2,seed=123)

'''Instantiate the classifier'''
clf = OneVsRestClassifier(LogisticRegression())

'''Fit the classifier to the training data'''
clf.fit(X_train, y_train)

print("Accuracy: {}".format(clf.score(X_test, y_test)))

<script.py> output:
    Accuracy: 0.0


## We want the probability of the prediction, not just 1 or 0 ##

'''Instantiate the classifier'''
clf = OneVsRestClassifier(LogisticRegression())

'''Fit it to the training data'''
clf.fit(X_train, y_train)

'''Load holdout data'''
holdout = pd.read_csv('HoldoutData.csv', index_col=0)

'''Generate predictions'''
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

'''Format predictions in DataFrame'''
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns, index=holdout.index, data=predictions)

'''Save to csv'''
prediction_df.to_csv('predictions.csv')

'''Submit predictions for scoring'''
score = score_submission(pred_path='predictions.csv')

print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))

#<script.py> output:
#    Your model, trained with numeric data only, yields logloss score: 1.9067227623381413
#
#    This is better than the benchmark score of 2.0455, which merely submitted uniform probabilities for each class

## Now add analysis of text data ##

'''Create a bag-of-words in scikit-learn'''

from sklearn.feature_extraction.text import CountVectorizer

'''Create the token pattern'''
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

'''Fill missing values in df.Position_Extra'''
df.Position_Extra.fillna('', inplace=True)

'''Instantiate the CountVectorizer'''
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

'''Fit to the data'''
vec_alphanumeric.fit(df.Position_Extra)

'''Print the number of tokens and first 15 tokens'''
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])

#<script.py> output:
#    There are 123 tokens in Position_Extra if we split on non-alpha numeric
#    ['1st', '2nd', '3rd', 'a', 'ab', 'additional', 'adm', 'administrative', 'and', 'any', 'art', 'assessment', 'assistant', 'asst', 'athletic']

## Combining text columns for tokenisation ##
#    * CountVectoriser() works by treating all entries in a column as single strings
#    * Therefore to look at several columns at once, will beed a method to turn a list of strings into a single string

def combine_text_columns(data_frame, to_drop = NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """

    '''Drop non-text columns that are in the df'''
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)

    '''Replace nans with blanks'''
    text_data.fillna('', inplace=True)

    '''Join all text items in a row that have a space in between'''
    return text_data.apply(lambda x: " ".join(x), axis=1)

## Compare results from tokenising with & without punctuation ##

from sklearn.feature_extraction.text import CountVectorizer

'''Create the basic token pattern'''
TOKENS_BASIC = '\\S+(?=\\s+)'

'''Create the alphanumeric token pattern'''
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

'''Instantiate basic CountVectorizer'''
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

'''Instantiate alphanumeric CountVectorizer'''
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

'''Create the text vector'''
text_vector = combine_text_columns(df)

'''Fit and transform vectorizers'''
vec_basic.fit_transform(text_vector)
vec_alphanumeric.fit_transform(text_vector)

# Print results
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))

#<script.py> output:
#    There are 1405 tokens in the dataset
#    There are 1117 alpha-numeric tokens in the dataset
#    As expected, there are fewer tokens when excluding punctuation

## Improving the model ##

## Instantiate pipeline - synthetic data ##

#sample_df.head()
#
#     numeric     text  with_missing label
#0 -10.856306               4.433240     b
#1   9.973454      foo      4.310229     b
#2   2.829785  foo bar      2.469828     a
#3 -15.062947               2.852981     b
#4  -5.786003  foo bar      1.826475     a


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

'''Split and select numeric data only, no nans .: only ‘numeric’ column, not ‘with_missing’ column'''
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],pd.get_dummies(sample_df['label']),random_state=22)

'''Instantiate Pipeline object'''
pl = Pipeline([('clf', OneVsRestClassifier(LogisticRegression()))])

'''Fit the pipeline to the training data'''
pl.fit(X_train, y_train)

'''Compute and print accuracy'''
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)

#<script.py> output:

#    Accuracy on sample data - numeric, no nans:  0.62

## Preprocessing numeric features ##

from sklearn.preprocessing import Imputer

'''Create training and test sets using only numeric data - include ‘with_missing’ this time as will preprocess using Imputer which replaces NaNs with mean value (by default)'''
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']],pd.get_dummies(sample_df['label']),random_state=456)

'''Insantiate Pipeline object'''
pl = Pipeline([('imp', Imputer()),('clf', OneVsRestClassifier(LogisticRegression()))])

'''Fit the pipeline to the training data'''
pl.fit(X_train, y_train)

'''Compute and print accuracy'''
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)

#<script.py> output:
#
#    Accuracy on sample data - all numeric, incl nans:  0.636


## Preprocessing text features ##
from sklearn.feature_extraction.text import CountVectorizer

'''Split out only the text data'''
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],pd.get_dummies(sample_df['label']),random_state=456)

'''Instantiate Pipeline object'''
pl = Pipeline([('vec', CountVectorizer()),('clf', OneVsRestClassifier(LogisticRegression()))])

'''Fit to the training data'''
pl.fit(X_train, y_train)

'''Compute and print accuracy'''
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)

#<script.py> output:
#
#    Accuracy on sample data - just text data:  0.808

## FunctionTransformer ##
#    * Any step in the pipeline must be an object that implements the fit and transform methods
#    * FunctionTransformer creates an object with these methods out of any Python function that you pass to it
#    * We will use this to help select subsets of data in a way that plays nicely with pipelines

from sklearn.preprocessing import FunctionTransformer

'''Obtain the data'''
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)

'''Fit and transform data'''
just_text_data = get_text_data.fit_transform(sample_df)
just_numeric_data = get_numeric_data.fit_transform(sample_df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())

#<script.py> output:
#    Text Data
#    0
#    1        foo
#    2    foo bar
#    3
#    4    foo bar
#    Name: text, dtype: object
#
#    Numeric Data
#         numeric  with_missing
#    0 -10.856306      4.433240
#    1   9.973454      4.310229
#    2   2.829785      2.469828
#    3 -15.062947      2.852981
#    4  -5.786003      1.826475


## Multiple types of processing: FeatureUnion ##
#    * FeatureUnion() allows separate preprocessing steps for different types of data
#    * It joins together two feature arrays (here the text feature array and the numeric feature array), to create a single array that can become input for a classifier

from sklearn.pipeline import FeatureUnion

'''Split using ALL data in sample_df'''
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],pd.get_dummies(sample_df['label']),random_state=22)

'''Create a FeatureUnion with nested pipeline'''
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )

'''Instantiate nested pipeline'''
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


'''Fit pl to the training data'''
pl.fit(X_train, y_train)

'''Compute and print accuracy'''
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)

#<script.py> output:
#
#    Accuracy on sample data - all data:  0.928


## Using FunctionTransformer on a the main dataset ##

from sklearn.preprocessing import FunctionTransformer

'''Get the dummy encoding of the labels'''
dummy_labels = pd.get_dummies(df[LABELS])

'''Get the columns that are features in the original df'''
NON_LABELS = [c for c in df.columns if c not in LABELS]

'''Split into training and test sets'''
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],dummy_labels,0.2,seed=123)

'''Preprocess data'''
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

'''Fit to the training data'''
pl.fit(X_train, y_train)

'''Compute and print accuracy'''
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

#<script.py> output:
#
#    Accuracy on budget dataset:  0.203846153846

## Try a different class of model ##
#    * Instead of LogisticRegression(), use RandomForestClassifier()

#<script.py> output:
#
#    Accuracy on budget dataset:  0.26153846153

## Can you adjust the model or parameters to improve accuracy? ##
#    * Try Random Forest with argument: n_estimators = 15
#<script.py> output:
#
#    Accuracy on budget dataset:  0.286538461538

## Adjustments in winning model ##

from sklearn.feature_extraction.text import CountVectorizer

'''Create the text vector'''
text_vector = combine_text_columns(X_train)

'''Create the token pattern'''
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

'''Instantiate the CountVectorizer'''
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

'''Fit text_features to the text vector'''
text_features.fit(text_vector)

'''Print the first 10 tokens'''
print(text_features.get_feature_names()[:10])

#<script.py> output:
#    ['00a', '12', '1st', '2nd', '3rd', '5th', '70', '70h', '8', 'a']

## N-gram range in scikit-learn ##
#    * ngram_range parameter sets the size of ngram features to be used in the model
#    * NB Steps in the pipeline not previously seen are to account for the fact that we are using a reduced-size sample of the original data
#        * dim_red = dimensionality reduction using chi2 test to select k ‘best’ features with SelectKBest()
#        * scale = scales features to between -1 and 1 using MaxAbsScaler()

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

'''Select 300 best features'''
chi_k = 300

'''Perform preprocessing'''
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

'''Create the token pattern'''
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

'''Instantiate pipeline'''
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

## Interaction terms ##

    pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

## Hashing ##

from sklearn.feature_extraction.text import HashingVectorizer

'''Get text data'''
text_data = combine_text_columns(X_train)

'''Create the token pattern'''
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

'''Instantiate the HashingVectorizer'''
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

'''Fit and transform the Hashing Vectorizer'''
hashed_text = hashing_vec.fit_transform(text_data)

'''Create DataFrame and print the head'''
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())

#<script.py> output:
#              0
#    0 -0.160128
#    1  0.160128
#    2 -0.480384
#    3 -0.320256
#    4  0.160128

#    * Although some text is hashed to the same value, this doesn’t materially impact performance in models of real-world problems

## Build the winning model ##

from sklearn.feature_extraction.text import HashingVectorizer

'''Instantiate the winning model pipeline'''
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC, non_negative=True, norm=None, binary=False, ngram_range=(1,2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

#    * Log loss: 1.2258
