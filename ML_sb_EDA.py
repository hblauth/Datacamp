import matplotlib.pyplot as plt

# Explore datatypes
df.dtypes.value_counts()

# Encode labels as categorical variables
# Define the lambda function
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)

# Explore distribution of unique labels
# Calculate number of unique values for each label
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')
plt.xlabel('Labels')
plt.ylabel('Number of unique values')
plt.show()
