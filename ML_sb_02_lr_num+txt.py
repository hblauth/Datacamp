# Analyse numeric and text

# Instantiate pipeline - synthetic data
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Split and select numeric data only, no nans .: only ‘numeric’ column, not ‘with_missing’ column
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],pd.get_dummies(sample_df['label']),random_state=22)

# Instantiate Pipeline object
pl = Pipeline([('clf', OneVsRestClassifier(LogisticRegression()))])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)

# Accuracy on sample data - numeric, no nans: 0.62

# Preprocessing numeric features
from sklearn.preprocessing import Imputer

# Create training and test sets using only numeric data - include ‘with_missing’ this time as will preprocess using Imputer which replaces NaNs with mean value (by default)
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']],pd.get_dummies(sample_df['label']),random_state=456)

# Instantiate Pipeline object
pl = Pipeline([('imp', Imputer()),('clf', OneVsRestClassifier(LogisticRegression()))])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)

# Accuracy on sample data - all numeric, incl nans: 0.636

# Preprocessing text features
from sklearn.feature_extraction.text import CountVectorizer

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],pd.get_dummies(sample_df['label']),random_state=456)

# Instantiate Pipeline object
pl = Pipeline([('vec', CountVectorizer()),('clf', OneVsRestClassifier(LogisticRegression()))])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)

# Accuracy on sample data - just text data: 0.808

# FunctionTransformer
#   Any step in the pipeline must be an object that implements the fit and transform methods
#   FunctionTransformer creates an object with these methods out of any Python function that you pass to it
#   Use this to help select subsets of data in a way that fits with pipelines

from sklearn.preprocessing import FunctionTransformer

# Obtain the data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)

# Fit and transform data
just_text_data = get_text_data.fit_transform(sample_df)
just_numeric_data = get_numeric_data.fit_transform(sample_df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())

# Multiple types of processing: FeatureUnion
#   FeatureUnion() allows separate preprocessing steps for different types of data
#   It joins together two feature arrays (here the text feature array and the numeric feature array), to create a single array that can become input for a classifier

from sklearn.pipeline import FeatureUnion

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],pd.get_dummies(sample_df['label']),random_state=22)

# Create a FeatureUnion with nested pipeline
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

# Instantiate nested pipeline
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)

# Accuracy on sample data - all data:  0.928

# Using FunctionTransformer on the main dataset

from sklearn.preprocessing import FunctionTransformer

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],dummy_labels,0.2,seed=123)

# Preprocess data
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

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)

# Accuracy on budget dataset:  0.203846153846
