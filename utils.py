import numpy as np

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

def recall_at_r(ranking, exact_ranking, r):
    """Compute the Recall@R."""
    return len(set(ranking[:r]) & set(exact_ranking[:r])) / r

def AP(ranking, exact_ranking):
    """Compute the Average Precision."""
    ap = 0
    num_rel = 0
    for i, idx in enumerate(ranking):
        if idx in exact_ranking:
            num_rel += 1
            ap += num_rel / (i+1)
    return ap / num_rel

def NMSE(original_data, decompressed_data):
    """Compute the Normalized Mean Squared Error."""
    squared_error = np.sum((original_data - decompressed_data)**2, axis=1)
    squared_norm = np.sum(np.square(original_data), axis=1)
    return np.mean(squared_error / squared_norm)