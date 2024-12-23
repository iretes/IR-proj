# %% [markdown]
# # GoVe Dataset

# %%
import pandas as pd
import csv
import pickle
RANDOM_SEED = 42

# %%
data = pd.read_table("./glove/glove.6B.300d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
data

# %%
data = data[data.index.notna()]
data = data[~data.index.str.contains(r'[^\w\s]', regex=True)]
data

# %%
tr_data = data.sample(n=25000, random_state=RANDOM_SEED)
remaining_data = data.drop(tr_data.index)
search_data = remaining_data.sample(n=10000, random_state=RANDOM_SEED)
remaining_data = remaining_data.drop(search_data.index)
queries = remaining_data.sample(n=1000, random_state=RANDOM_SEED)

tr_data_array = tr_data.to_numpy()
search_data_array = search_data.to_numpy()
queries_array = queries.to_numpy()

with open('./glove/glove_data.pkl', 'wb') as f:
    pickle.dump((tr_data_array, search_data_array, queries_array), f)

# %%
search_data = tr_data.sample(n=10000, random_state=RANDOM_SEED)
search_data = search_data.to_numpy()

with open('./glove/glove_data_overlap.pkl', 'wb') as f:
    pickle.dump((tr_data_array, search_data_array, queries_array), f)


