# %% [markdown]
# # GoVe Dataset

# %% [markdown]
# Importing the libraries:

# %%
import pandas as pd
import csv
import pickle
RANDOM_SEED = 42

# %% [markdown]
# Loading the dataset:

# %%
data = pd.read_table("./glove/glove.6B.300d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
data

# %% [markdown]
# Excluding punctuation:

# %%
data = data[data.index.notna()]
data = data[~data.index.str.contains(r'[^\w\s]', regex=True)]
data

# %% [markdown]
# Subsampling 25000 samples for the training set, 10000 samples for the search set and 1000 samples for the query set. The data is converted to numpy arrays and saved to disk. In the file "./glove/glove_data.pkl", the sets are mutually exclusive, whereas in "./glove/glove_data_overlap.pkl", the search set is a subset of the training set

# %%
tr_data = data.sample(n=25000, random_state=RANDOM_SEED)
remaining_data = data.drop(tr_data.index)
search_data = remaining_data.sample(n=10000, random_state=RANDOM_SEED)
remaining_data = remaining_data.drop(search_data.index)
queries = remaining_data.sample(n=1000, random_state=RANDOM_SEED)

tr_data_array = tr_data.to_numpy()
search_data_array = search_data.to_numpy()
queries_array = queries.to_numpy()

# saving the data (no overlap between training, search and queries sets)
with open('./glove/glove_data.pkl', 'wb') as f:
    pickle.dump((tr_data_array, search_data_array, queries_array), f)

search_data = tr_data.sample(n=10000, random_state=RANDOM_SEED)
search_data = search_data.to_numpy()

# saving the data (overlap between training and search sets)
with open('./glove/glove_data_overlap.pkl', 'wb') as f:
    pickle.dump((tr_data_array, search_data_array, queries_array), f)  # TODO: valuta se togliere


