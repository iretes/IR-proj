import csv
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_descriptors(name, dir):
    xt = fvecs_read(f"{dir}/{name}_learn.fvecs")
    xb = fvecs_read(f"{dir}/{name}_base.fvecs")
    xq = fvecs_read(f"{dir}/{name}_query.fvecs")
    gt = ivecs_read(f"{dir}/{name}_groundtruth.ivecs")

    return xb, xq, xt, gt

def load_data(dataset_name, dataset_dir, search_train_subset=False, random_seed=None):
    """
    Load the data for the specified dataset.

    Parameters
    ----------

    dataset_name : str
        Name of the dataset to load. Choose from 'siftsmall', 'sift', 'gist' or
        'glove'.

    dataset_dir : str
        Directory where the dataset files are stored.

    search_train_subset : bool, optional
        If True, the search data will be a subset of the training data. Default
        is False.

    random_seed : int, optional
        Random seed to use for reproducibility. Default is None.

    Returns
    -------

    tr_data : numpy.ndarray
        Training data.

    search_data : numpy.ndarray
        Search data.

    queries : numpy.ndarray
        Query data.

    gt : numpy.ndarray
        Ground truth data. Only returned for 'siftsmall', 'sift' and 'gist'
        datasets.
    
    """
    gt = None
    rng = np.random.default_rng(random_seed)
    if dataset_name in ["siftsmall", "sift", "gist"]:
        search_data, queries, tr_data, gt = load_descriptors(name=dataset_name,
            dir=dataset_dir)
        if search_train_subset:
            search_data = tr_data[rng.choice(tr_data.shape[0], 10000, replace=False)]
        if dataset_name == "gist":
            # subsample data
            search_data = search_data[rng.choice(search_data.shape[0], 10000, replace=False)]
            queries = queries[rng.choice(queries.shape[0], 100, replace=False)]
            tr_data = tr_data[rng.choice(tr_data.shape[0], 25000, replace=False)]
    elif dataset_name == "glove":
        data = pd.read_table(f"{dataset_dir}/glove.6B.300d.txt", sep=" ",
            index_col=0, header=None, quoting=csv.QUOTE_NONE)
        # excluding missing values, non-alphanumeric characters and punctuation
        data = data[data.index.notna()]
        data = data[~data.index.str.contains(r'[^\w\s]', regex=True)]
        # subsample data
        tr_data = data.sample(n=25000, random_state=random_seed, replace=False)
        remaining_data = data.drop(tr_data.index)
        queries = remaining_data.sample(n=1000, random_state=random_seed, replace=False)
        remaining_data = remaining_data.drop(queries.index)
        if search_train_subset:
            search_data = tr_data.sample(n=10000, random_state=random_seed, replace=False)
            search_data = search_data.to_numpy()
        else:
            search_data = remaining_data.sample(n=10000, random_state=random_seed, replace=False)
        tr_data = tr_data.to_numpy()
        search_data = search_data.to_numpy()
        queries = queries.to_numpy()
        # normalize data to unit vectors
        for i in range(tr_data.shape[0]):
            tr_data[i] = tr_data[i] / np.linalg.norm(tr_data[i])
        for i in range(search_data.shape[0]):
            search_data[i] = search_data[i] / np.linalg.norm(search_data[i])
        for i in range(queries.shape[0]):
            queries[i] = queries[i] / np.linalg.norm(queries[i])
    else:
        raise ValueError("Invalid dataset name. Choose from 'siftsmall', 'sift', 'gist' or 'glove'.")

    return tr_data, search_data, queries, gt

def recall_at_r(ranking, exact_ranking, r):
    """
    Compute the Recall@R.

    Parameters
    ----------

    ranking : list or numpy.ndarray
        Ranking of the elements.

    exact_ranking : list or numpy.ndarray
        Exact ranking of the elements.

    r : int
        Number of elements to consider.

    Returns
    -------

    recall : float
        Recall@R.
    
    """
    return len(set(ranking[:r]) & set(exact_ranking[:r])) / r

def NMSE(original_data, decompressed_data):
    """
    Compute the Normalized Mean Squared Error.

    Parameters
    ----------

    original_data : numpy.ndarray
        Original data.

    decompressed_data : numpy.ndarray
        Decompressed data.

    Returns
    -------

    nmse : float
        Normalized Mean Squared Error.
    
    """
    squared_error = np.sum((original_data - decompressed_data)**2, axis=1)
    squared_norm = np.sum(np.square(original_data), axis=1)
    return np.mean(squared_error / squared_norm)

def sankey_plot(
        labels,
        labels_titles=None,
        title=None,
        color_palette=sns.color_palette()
    ):
    """
    Plots a Sankey diagram of the sets of labels passed as arguments.

    Parameters
    ----------

    labels : list of numpy.ndarray
        List of labels for each set of data.

    labels_titles : list of str, optional
        List of titles for each set of labels. Default is None.

    title : str, optional
        Title of the plot. Default is None.

    color_palette : seaborn color palette, optional
        Color palette to use for the plot. Default is seaborn color palette.
    
    """

    n_clusters = [len(set(label_list)) for label_list in labels]

    plot_labels = []
    for i in range(len(labels)):
        for j in range(n_clusters[i]):
            plot_labels.append(str(j))

    source = []
    target = []
    value = []
    for i in range(len(labels)-1):
        confusion_matrix = pd.crosstab(labels[i], labels[i+1])
        curr_source = []
        curr_target = []
        curr_value = []

        source_add = 0
        for j in range(0, i):
            source_add += n_clusters[j]
        target_add = source_add + n_clusters[i]

        for j in range(n_clusters[i]):
            for k in range(n_clusters[i+1]):
                if confusion_matrix.iloc[j, k] != 0:
                    curr_source.append(j+source_add)
                    curr_target.append(k+target_add)
                    curr_value.append(confusion_matrix.iloc[j, k])

        source += curr_source
        target += curr_target
        value += curr_value

    colors = []
    for i in range(len(labels)):
        colors += color_palette.as_hex()[:n_clusters[i]]

    fig = go.Figure(
        data=[
            go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = "black", width = 0.5),
                    label = plot_labels,
                    color = colors
                ),
                link = dict(
                    source = source,
                    target = target,
                    value = value
                )
            )
        ]
    )

    for x_coordinate, column_name in enumerate(labels_titles):
        fig.add_annotation(
            x=x_coordinate,
            y=1.05,
            xref="x",
            yref="paper",
            text=column_name,
            showarrow=False
        )
    fig.update_layout(
        title_text=title, 
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        plot_bgcolor='rgba(0,0,0,0)',
        font_size=10
    )
    
    fig.show()