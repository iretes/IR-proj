import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_sift(name="siftsmall", dir="siftsmall"):
    xt = fvecs_read(f"{dir}/{name}_learn.fvecs")
    xb = fvecs_read(f"{dir}/{name}_base.fvecs")
    xq = fvecs_read(f"{dir}/{name}_query.fvecs")
    gt = ivecs_read(f"{dir}/{name}_groundtruth.ivecs")

    return xb, xq, xt, gt

def NDCG(ranking, exact_ranking):
    """Compute the Normalized Discounted Cumulative Gain."""
    dcg = 0
    idcg = 0
    for i, idx in enumerate(ranking):
        if idx in exact_ranking:
            dcg += 1 / np.log2(i+2)
        if i < len(exact_ranking):
            idcg += 1 / np.log2(i+2)
    return dcg / idcg

def recall_at_r(ranking, exact_ranking, r): # TODO: è analogo a precision?
    """Compute the Recall@R."""
    return len(set(ranking[:r]) & set(exact_ranking[:r])) / r

def AP_at_r(ranking, exact_ranking, r): # TODO: utile?
    """Compute the Average Precision@R."""
    ap = 0
    num_rel = 0
    for i, idx in enumerate(ranking[:r]):
        if idx in exact_ranking[:r]:
            num_rel += 1
            ap += num_rel / (i+1)
    return ap / num_rel

def NMSE(original_data, decompressed_data):
    """Compute the Normalized Mean Squared Error."""
    squared_error = np.sum((original_data - decompressed_data)**2, axis=1)
    squared_norm = np.sum(np.square(original_data), axis=1)
    return np.mean(squared_error / squared_norm)

def sankey_plot(
        labels,
        labels_titles=None,
        title=None,
        color_palette=sns.color_palette()
    ):
    """Plots a Sankey diagram of the sets of labels passed as arguments."""

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

    # file_name = f'../html/sankey'
    # if title is not None:
    #     camel_title = title.replace(' ', '_')
    #     file_name += f'_{camel_title}'
    # file_name += '.html'
    # pyo.plot(fig, filename=file_name, auto_open=False)
    fig.show()