# Analysis of text data

# Create a bag-of-words in scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

# Create the token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Fill missing values in df.Position_Extra
df.Position_Extra.fillna('', inplace=True)

#Instantiate the CountVectorizer
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

#Fit to the data
vec_alphanumeric.fit(df.Position_Extra)

#Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])

# Combining text columns for tokenisation
#   CountVectoriser() treats all entries in a column as single strings
#   Therefore to look at several columns at once, will need a method to turn a list of strings into a single string

def combine_text_columns(data_frame, to_drop = NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """

    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)

    # Replace nans with blanks
    text_data.fillna('', inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

# Compare results from tokenising with & without punctuation

from sklearn.feature_extraction.text import CountVectorizer

# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)'

# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate basic CountVectorizer
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

# Instantiate alphanumeric CountVectorizer
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Create the text vector
text_vector = combine_text_columns(df)

# Fit and transform vectorizers
vec_basic.fit_transform(text_vector)
vec_alphanumeric.fit_transform(text_vector)

# Print results
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))
