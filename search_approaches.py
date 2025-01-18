import numpy as np
from scipy.cluster.vq import vq
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import (KMeans, BisectingKMeans, MiniBatchKMeans)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skfda import FDataGrid
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from skfda.ml.clustering import FuzzyCMeans

class PQ:
    """
    Product Quantization (PQ) implementation.
    """

    M: int
    """
    Number of subspaces.
    """
    K: int
    """
    Number of clusters per subspace.
    """
    kmeans_iter: int
    """
    Maximum number of iterations for KMeans.
    """
    kmeans_minit: str
    """
    Method for KMeans initialization.
    """
    seed: int
    """
    Random seed.
    """
    orth_transf: bool
    """
    Apply orthogonal transformation to the data.
    """
    dim_reduction: bool
    """
    Apply PCA transformation to reduce dimensionality of each subspace.
    """
    shrink_threshold: float
    """
    Threshold for shrinking centroids to remove features.
    """
    Ds: int
    """
    Dimension of each subspace when using equal partitioning.
    """
    D: int
    """
    Number of features.
    """
    codebook: list[np.ndarray]
    """
    Cluster centroids for each subspace.
    """
    pqcode: np.ndarray
    """
    Quantized representation of the data added to the database.
    """
    avg_dist: np.ndarray
    """
    Average distortion (average squared euclidean distance from the centroid)
    for each cluster in each subspace of the data added to the database.
    """
    inertia: np.ndarray
    """
    Inertia of the KMeans clustering in each subspace (sum of squared distances
    of samples to their closest cluster center, weighted by the sample weights
    if provided).
    """
    features_labels: np.ndarray
    """
    Cluster labels of the features for partitioning.
    """
    features_cluster_sizes: np.ndarray
    """
    Number of features in each subspace.
    """
    energy_std: np.ndarray
    """
    Average energy (the sum of squared components) within each subspace.
    """

    def __init__(self, M: int = 8, K: int = 256, kmeans_iter: int = 300,
        kmeans_minit: str = "k-means++", seed: int = None,
        orth_transf: bool = False, dim_reduction: bool = False,
        shrink_threshold: float = None):
        """
        Constructor.

        Parameters
        ----------

        M : int, default=8
            Number of subspaces.
        
        K : int, default=256
            Number of clusters per subspace.
        
        kmeans_iter : int, default=300
            Maximum number of iterations for KMeans.
        
        kmeans_minit : str, default='k-means++'
            Method for KMeans initialization.
            See https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html
        
        seed : int, default None
            Random seed.
        
        orth_transf : bool, default=False
            Apply orthogonal transformation to the data.

        dim_reduction : bool, default=False
            Apply PCA transformation to reduce dimensionality of each subspace.

        shrink_threshold : float, default=None
            Threshold for shrinking centroids to remove features.

        """

        if M <= 0:
            raise ValueError("M must be greater than 0.")
        if K <= 0:
            raise ValueError("K must be greater than 0.")
        if shrink_threshold and shrink_threshold < 0:
            raise ValueError("Shrink threshold must be greater or equal to 0.")

        self.M = M
        self.K = K
        self.kmeans_iter = kmeans_iter
        self.kmeans_minit = kmeans_minit
        self.seed = seed
        self.orth_transf = orth_transf
        self.dim_reduction = dim_reduction
        self.shrink_threshold = shrink_threshold

        K_bits = np.log2(self.K-1)
        if K_bits <= 8:
            self.code_inttype = np.uint8
        elif K_bits <= 16:
            self.code_inttype = np.uint16
        elif K_bits <= 32:
            self.code_inttype = np.uint32
        else:
            self.code_inttype = np.uint64

        self.Ds = None
        self.D = None
        self.codebook = None
        self.pqcode = None
        self.avg_dist = None
        self.inertia = None
        self.features_labels = None
        self.features_cluster_sizes = None
        self._chunk_start = None
        self._features_perm = None
        self._pcas = None
        self._O = None
        self._ncs = None
        self.energy_std = None
        
    def _compute_partitions(self, data: np.ndarray,
        features_labels: np.ndarray = None) -> None:
        """
        
        Compute the partitioning of the features into subspaces.

        Parameters
        ----------

        data : np.ndarray
            Data to partition.

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces.
        
        """
        
        if features_labels is None:
            self.features_cluster_sizes = np.full(self.M, self.Ds)
            self._chunk_start = np.arange(0, self.M * self.Ds + self.Ds, self.Ds)
            self._features_perm = np.arange(data.shape[1])  # identity permutation
        else:
            self.features_labels = features_labels
            _, self.features_cluster_sizes = np.unique(self.features_labels,
                return_counts=True)
            self._chunk_start = np.zeros(self.M+1, dtype=int)
            self._chunk_start[1:] = np.cumsum(self.features_cluster_sizes)
            self._features_perm = np.argsort(self.features_labels)

    def _neighbor_distances_to_weights(self, distances: np.ndarray,
        inverse_weights: bool, weight_method: str) -> np.ndarray:
        """
        Convert distances to the neighbor-th nearest neighbor to sample weights
        for KMeans clustering of subspaces.

        Parameters
        ----------

        distances : np.ndarray
            Distances to the neighbor-th nearest neighbor for each vector.

        inverse_weights : bool
            If True, the weights are inversely proportional to the distance to
            the neighbor-th nearest neighbor.

        weight_method : str
            Method for computing weights:
            * 'normal': Normalize the distances to [0, 1].
            * 'reciprocal': If `inverse_weights` is True, compute the reciprocal
                of the distances and normalize to [0, 1], otherwise normalize
                the distances to [0, 1].

        Returns
        -------

        weights : np.ndarray
            Weights for KMeans clustering of subspaces.
        
        """

        if weight_method == "normal":
            weights = distances / np.max(distances)
            if inverse_weights:
                weights = 1 - weights
        else:
            distances = np.where(distances == 0, np.finfo(float).eps, distances)
            if inverse_weights:
                weights = 1 / distances
            else:
                weights = distances
            weights = weights / np.max(weights)
        
        return weights

    def _compute_clustering_weights(self, data: np.ndarray,
        neighbor: int, inverse_weights: bool, weight_method: str) -> np.ndarray:
        """
        Compute weights for KMeans clustering based on the distance to the
        neighbor-th nearest neighbor.

        Parameters
        ----------

        data : np.ndarray
            Data to compute the weights.

        neighbor : int
            Neighbor-th nearest neighbor for weighting samples.

        inverse_weights : bool
            If True, the weights are inversely proportional to the distance to
            the neighbor-th nearest neighbor.

        weight_method : str
            Method for computing weights:
            * 'normal': Normalize the distances to [0, 1].
            * 'reciprocal': If `inverse_weights` is True, compute the reciprocal
                of the distances and normalize to [0, 1], otherwise normalize
                the distances to [0, 1].

        Returns
        -------

        weights : np.ndarray
            Weights for KMeans clustering.
        
        """
        
        knn = NearestNeighbors(n_neighbors=neighbor).fit(data)
        distances, _ = knn.kneighbors(data)
        weights = self._neighbor_distances_to_weights(distances[:, -1],
            inverse_weights, weight_method)
        return weights

    def train(self, data: np.ndarray, add: bool = True,
        compute_distortions: bool = False, compute_energy: bool = False,
        features_labels: np.ndarray = None, num_dims: int = None,
        whiten: bool = False, weight_samples: bool = False, neighbor: int = 3,
        inverse_weights: bool = True, weight_method: str = "normal", 
        verbose: bool = False) -> None:
        """
        Train the quantizer on the given data.
        
        Parameters
        ----------

        data : np.ndarray
            Data to train the quantizer.

        add : bool, default=True
            Add the data to the database.

        compute_distortions : bool, default=False
            Compute the average distortion for each cluster in each subspace
            (if `add` is also True).

        compute_energy : bool, default=False
            Compute the average energy (the sum of squared components) within
            each subspace.

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces.

        num_dims : int, default=None
            Number of dimensions in each subspace after PCA dimensionality
            reduction. If `self.dim_reduction` is False, but `num_dims` is
            provided, centroids are computed in the reduced space and then
            transformed back to the original space.

        whiten : bool, default=False
            If True, apply whitening to the PCA transformation.

        weight_samples : bool, default=False
            Weight samples while training KMeans based on the distance to
            the neighbor-th nearest neighbor.

        neighbor : int, default=3
            Neighbor-th nearest neighbor for weighting samples (if
            `weight_samples` is True).

        inverse_weights : bool, default=True
            If True, the weights are inversely proportional to the distance to
            the neighbor-th nearest neighbor (when `weight_samples` is True).

        weight_method : str, default='normal'
            Method for computing weights (when weight_samples is True):
            * 'normal': Normalize the distances to [0, 1].
            * 'reciprocal': If `inverse_weights` is True, compute the reciprocal
                of the distances and normalize to [0, 1], otherwise normalize
                the distances to [0, 1].

        verbose : bool, default=False
            Print training information.

        """

        self.D = data.shape[1]
        if self.D % self.M != 0:
            raise ValueError("Feature dimension must be divisible by the number"
                " of subspaces (M).")
        if features_labels is not None and features_labels.shape[0] != self.D:
            raise ValueError("Feature labels must have the same number of"
                " features as the data.")
        if self.dim_reduction and num_dims is None:
            raise ValueError("Number of dimensions must be provided for"
                " dimensionality reduction.")
        if neighbor < 1:
            raise ValueError("Neighbor must be greater than 0.")
        if weight_method not in ["normal", "reciprocal"]:
            raise ValueError("Supported weight methods are 'normal' and"
                " 'reciprocal'.")
        if weight_samples and self.shrink_threshold:
            raise Warning("Shrink threshold will be applied on unweighted"
                " centroids.")

        self.Ds = int(self.D / self.M)
        self.codebook = []
        self.pqcode = None
        self.avg_dist = None
        self.inertia = np.empty((self.M))

        if self.shrink_threshold:
            self._ncs = []
        
        if add:
            self.pqcode = np.empty((data.shape[0], self.M), self.code_inttype)
            if compute_distortions:
                self.avg_dist = np.zeros((self.M, self.K), np.float32)
        
        if self.orth_transf:
            self._O = ortho_group.rvs(dim=self.D)
            data = np.dot(data, self._O)
        
        if self.dim_reduction:
            self._pcas = []

        self._compute_partitions(data, features_labels)
        if num_dims and self.features_cluster_sizes.min() < num_dims:
            raise ValueError("Number of dimensions for dimensionality reduction"
                " must be less than the number of features in each subspace.")
        data = data[:, self._features_perm]

        if add and compute_energy:
            self.energy_std = self.compute_mean_energy(data,
                compute_partitions=False)

        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            
            sample_weight = None
            if weight_samples:
                sample_weight = self._compute_clustering_weights(data_sub,
                    neighbor, inverse_weights, weight_method)
            
            km = KMeans(n_clusters=self.K, init=self.kmeans_minit, n_init=1,
                random_state=self.seed, max_iter=self.kmeans_iter)
            
            if num_dims:
                pca = PCA(n_components=num_dims, whiten=whiten).fit(data_sub)
                data_sub_red = pca.transform(data_sub)
                km = km.fit(data_sub_red, sample_weight=sample_weight)
                if self.dim_reduction:
                    self._pcas.append(pca)
                else:
                    km.cluster_centers_ = pca.inverse_transform(km.cluster_centers_)
            else:
                km = km.fit(data_sub, sample_weight=sample_weight)

            self.inertia[m] = km.inertia_
            
            if verbose:
                print(f"KMeans on subspace {m+1} converged in {km.n_iter_}"
                    f" iterations with an inertia of {km.inertia_}.")

            if self.shrink_threshold:
                nc = NearestCentroid(shrink_threshold=self.shrink_threshold)
                nc = nc.fit(data_sub, km.labels_)
                self._ncs.append(nc)
                self.codebook.append(nc.centroids_)
            else:
                self.codebook.append(km.cluster_centers_)

            if add:
                self.pqcode[:, m], _ = vq(data_sub, self.codebook[m])
                if compute_distortions:
                    for k in range(self.K):
                        dist = cdist(data_sub[self.pqcode[:, m] == k],
                            [self.codebook[m][k]], 'sqeuclidean')
                        self.avg_dist[m, k] = np.mean(dist)

    def add(self, data: np.ndarray, compute_distortions: bool = False,
        compute_energy: bool = False) -> None:
        """
        Add data to the database.

        Parameters
        ----------

        data : np.ndarray
            Data to add to the database.
        
        compute_distortions : bool, default=False
            Compute the average distortion for each cluster in each subspace.

        compute_energy : bool, default=False
            Compute the average energy (the sum of squared components) within
            each subspace.

        Notes
        ----

        If called multiple times, the previous data is lost.

        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before adding data.")
        if data.shape[1] != self.D:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        self.pqcode = self.compress(data)

        if self.features_labels is not None:
            data = data[:, self._features_perm]

        if compute_energy:
            self.energy_std = self.compute_mean_energy(data,
                compute_partitions=False)

        if compute_distortions:
            self.avg_dist = np.zeros((self.M, self.K), np.float32)
            for m in range(self.M):
                data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
                for k in range(self.K):
                    dist = cdist(data_sub[self.pqcode[:, m] == k],
                        [self.codebook[m][k]], 'sqeuclidean')
                    self.avg_dist[m, k] = np.mean(dist)

    def compress(self, data: np.ndarray) -> np.ndarray:
        """
        Compress data using the trained quantizer.

        Parameters
        ----------

        data : np.ndarray
            Data to compress.

        Returns
        -------

        compressed : np.ndarray
            Compressed representation of the data.
        
        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before"
                " compressing.")
        if data.shape[1] != self.D:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        if self.orth_transf:
            data = np.dot(data, self._O)

        if self.features_labels is not None:
            data = data[:, self._features_perm]

        compressed = np.empty((data.shape[0], self.M), self.code_inttype)
        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            if self.dim_reduction:
                data_sub = self._pcas[m].transform(data_sub)
            compressed[:, m], _ = vq(data_sub, self.codebook[m])
        
        return compressed
    
    def decompress(self, codes: np.ndarray) -> np.ndarray:
        """
        Decompress codes using the trained quantizer.

        Parameters
        ----------

        codes : np.ndarray
            Codes to decompress.

        Returns
        -------

        decompressed : np.ndarray
            Decompressed representation of the codes.
        
        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before"
                " decompressing.")
        if codes.shape[1] != self.M:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        decompressed = np.empty((codes.shape[0], self.D), np.float32)
        for m in range(self.M):
            if self.dim_reduction:
                decompressed[:, self._chunk_start[m] : self._chunk_start[m+1]] = \
                    self._pcas[m].inverse_transform(self.codebook[m][codes[:, m]])
            else:
                decompressed[:, self._chunk_start[m] : self._chunk_start[m+1]] = \
                    self.codebook[m][codes[:, m]]
        
        if self.features_labels is not None:
            decompressed = decompressed[:, np.argsort(self._features_perm)]

        if self.orth_transf:
            decompressed = np.dot(decompressed, self._O.T)
        
        return decompressed
    
    def search(self, query: np.ndarray, subset: np.ndarray = None,
        asym: bool = True, correct: bool = False, sort: bool = True) -> \
        tuple[np.ndarray, np.ndarray]:
        """
        Compute the distances of the query to the database vectors.

        Parameters
        ----------

        query : np.ndarray
            Query vector.

        subset : np.ndarray, default=None
            Indices of the database vectors to search in.
        
        asym : bool, default=True
            Use asymmetric distance computation (do not quantize the query).

        correct : bool, default=False
            Correct distances by adding average distortions.

        sort : bool, default=True
            Sort the distances returned.

        Returns
        -------

        dists : np.ndarray
            Distances of the query to the database vectors.
        
        idx : np.ndarray
            Indices of the database vectors sorted by distance in increasing
            order, if `sort` is True.
        
        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before searching.")
        if self.pqcode is None:
            raise ValueError("Vectors must be added before searching.")
        if correct and self.avg_dist is None:
            raise ValueError("Distorsion must be computed before correcting.")
        if len(query) != self.D:
            raise ValueError("Query dimensions must match trained data"
                " dimensions.")
        if subset is not None and subset.shape[0] > self.pqcode.shape[0]:
            raise ValueError("Subset size must be less or equal to the number"
                " of added vectors.")
        if subset is not None and \
            (subset.max() >= self.pqcode.shape[0] or subset.min() < 0):
            raise ValueError("Subset indices must be non-negative and less than"
                " the number of added vectors.")

        if subset is None:
            subset = slice(None)

        if self.orth_transf:
            query = np.dot(query, self._O)

        if self.features_labels is not None:
            query = query[self._features_perm]

        dist_table = np.empty((self.M, self.K), np.float32)
        for m in range(self.M):
            query_sub = query[self._chunk_start[m] : self._chunk_start[m+1]]
            if self.dim_reduction:
                query_sub = self._pcas[m].transform([query_sub]).reshape(-1)
            if not asym:
                query_sub_code, _ = vq([query_sub], self.codebook[m])
                query_sub = self.codebook[m][query_sub_code[0]]
            dist_table[m, :] = cdist([query_sub], self.codebook[m],
                'sqeuclidean')[0]
            if correct:
                dist_table[m, :] += self.avg_dist[m]
                if not asym:
                    dist_table[m, query_sub_code] += self.avg_dist[m, query_sub_code]

        dist = np.sum(dist_table[range(self.M), self.pqcode[subset]], axis=1)
        
        if sort:
            return dist, np.argsort(dist)
        return dist, None
    
    def plot_neighbor_distances(self, data: np.ndarray, neighbor: int,
        ax: plt.Axes, features_labels: np.ndarray = None) -> None:
        """
        Plot the distances to the neighbor-th nearest neighbor for each
        vector in the dataset in each subspace.

        Parameters
        ----------

        data : np.ndarray
            Data to plot the distances.

        neighbor : int
            Neighbor-th nearest neighbor for computing distances.

        ax : plt.Axes
            Axes to plot the distances

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces.
        
        """
        
        self.D = data.shape[1]
        if self.D % self.M != 0:
            raise ValueError("Feature dimension must be divisible by the number"
                " of subspaces (M).")
        if features_labels is not None and features_labels.shape[0] != self.D:
            raise ValueError("Feature labels must have the same number of"
                " features as the data.")
        self.Ds = int(self.D / self.M)

        self._compute_partitions(data, features_labels)
        data = data[:, self._features_perm]

        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            knn = NearestNeighbors(n_neighbors=neighbor).fit(data_sub)
            distances, _ = knn.kneighbors(data_sub)
            ax.plot(np.sort(distances[:, -1]), label=f"Subspace {m+1}")
        
        ax.set_ylabel(f"{neighbor}-th nearest neighbor distance")
        ax.set_xlabel("Vectors")
        ax.legend()
    
    def plot_variance_explained(self, data: np.ndarray,
        features_labels: np.ndarray = None) -> None:
        """
        Plot the variance explained by each principal component for each
        subspace.

        Parameters
        ----------

        data : np.ndarray
            Data to plot the variance explained.

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces.
        
        """
        
        self.D = data.shape[1]
        if self.D % self.M != 0:
            raise ValueError("Feature dimension must be divisible by the number"
                " of subspaces (M).")
        if features_labels is not None and features_labels.shape[0] != self.D:
            raise ValueError("Feature labels must have the same number of"
                " features as the data.")
        self.Ds = int(self.D / self.M)

        self._compute_partitions(data, features_labels)
        data = data[:, self._features_perm]

        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            pca = PCA().fit(data_sub)
            plt.plot(pca.explained_variance_ratio_, label=f"Subspace {m+1}")
        
        plt.ylabel("Variance explained")
        plt.xlabel("Principal components")
        plt.legend()

    def compute_mean_energy(self, data: np.ndarray,
        compute_partitions: bool = True, features_labels: np.ndarray = None
        ) -> None:
        """
        Compute the average energy (the sum of squared components) within each
        subspace.

        Parameters
        ----------

        data : np.ndarray
            Data to compute the energy.

        compute_partitions : bool, default=True
            Compute the partitioning of the features into subspaces.

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces.

        Returns
        -------

        energy_std : np.ndarray
            Average energy in each subspace.
        
        """

        if compute_partitions:
            self.D = data.shape[1]
            if self.D % self.M != 0:
                raise ValueError("Feature dimension must be divisible by the number"
                    " of subspaces (M).")
            if features_labels is not None and features_labels.shape[0] != self.D:
                raise ValueError("Feature labels must have the same number of"
                    " features as the data.")
            self.Ds = int(self.D / self.M)

            self._compute_partitions(data, features_labels)
            data = data[:, self._features_perm]

        energy_std = np.zeros((self.M))
        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            energy = np.sum(data_sub ** 2, axis=1)
            energy_std[m] = np.mean(energy)
        return energy_std

class FuzzyPQ(PQ):
    """
    Fuzzy Product Quantization implementation.
    """

    M: int
    """
    Number of subspaces.
    """
    K: int
    """
    Number of clusters per subspace.
    """
    kmeans_iter: int
    """
    Maximum number of iterations for Fuzzy KMeans.
    """
    seed: int
    """
    Random seed.
    """
    orth_transf: bool
    """
    Apply orthogonal transformation to the data.
    """
    dim_reduction: bool
    """
    Apply PCA transformation to reduce dimensionality of each subspace.
    """
    Ds: int
    """
    Dimension of each subspace when using equal partitioning.
    """
    D: int
    """
    Number of features.
    """
    codebook: list[np.ndarray]
    """
    Cluster centroids for each subspace.
    """
    pqcode: np.ndarray
    """
    Quantized representation of the data added to the database (codes of the two
    clusters with highest membership).
    """
    fuzzifier: float
    """
    Hyper-parameter that controls how fuzzy the cluster will be.
    """
    membership_ratio: np.ndarray
    """
    Membership ratio of the two clusters with highest membership (second highest
    membership / highest membership) for the data added to the database.
    """
    inertia: np.ndarray
    """
    Inertia of the Fuzzy KMeans clustering in each subspace (sum of squared
    distances of samples to their closest cluster center, weighted by the sample
    weights if provided).
    """
    features_labels: np.ndarray
    """
    Cluster labels of the features for partitioning.
    """
    features_cluster_sizes: np.ndarray
    """
    Number of features in each subspace.
    """
    energy_std: np.ndarray
    """
    Average energy (the sum of squared components) within each subspace.
    """

    def __init__(self, M: int = 8, K: int = 256, kmeans_iter: int = 300,
        fuzzifier: float = 2, seed: int = None, orth_transf: bool = False,
        dim_reduction: bool = False):
        """
        Constructor.

        Parameters
        ----------

        M : int, default=8
            Number of subspaces.
        
        K : int, default=256
            Number of clusters per subspace.
        
        kmeans_iter : int, default=300
            Maximum number of iterations for Fuzzy KMeans.

        fuzzifier : float, default=2
            Hyper-parameter that controls how fuzzy the cluster will be.
            The higher it is, the fuzzier the cluster will be in the end.
            This parameter should be greater than 1.
        
        seed : int, default None
            Random seed.
        
        orth_transf : bool, default=False
            Apply orthogonal transformation to the data.

        dim_reduction : bool, default=False
            Apply PCA transformation to reduce dimensionality of each subspace.

        """

        super().__init__(M=M, K=K, kmeans_iter=kmeans_iter, seed=seed,
            orth_transf=orth_transf, dim_reduction=dim_reduction)
        self.fuzzifier = fuzzifier
        self.membership_ratio = None

    def train(self, data: np.ndarray, add: bool = True,
        compute_energy: bool = False, features_labels: np.ndarray = None,
        num_dims: int = None, whiten: bool = False,
        verbose: bool = False) -> None:
        """
        Train the quantizer on the given data.
        
        Parameters
        ----------

        data : np.ndarray
            Data to train the quantizer.

        add : bool, default=True
            Add the data to the database.

        compute_energy : bool, default=False
            Compute the average energy (the sum of squared components) within
            each subspace.

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces.

        num_dims : int, default=None
            Number of dimensions in each subspace after PCA dimensionality
            reduction. If `self.dim_reduction` is False, but `num_dims` is
            provided, centroids are computed in the reduced space and then
            transformed back to the original space.

        whiten : bool, default=False
            If True, apply whitening to the PCA transformation.

        verbose : bool, default=False
            Print training information.

        """

        self.D = data.shape[1]
        if self.D % self.M != 0:
            raise ValueError("Feature dimension must be divisible by the number"
                " of subspaces (M).")
        if features_labels is not None and features_labels.shape[0] != self.D:
            raise ValueError("Feature labels must have the same number of"
                " features as the data.")
        if self.dim_reduction and num_dims is None:
            raise ValueError("Number of dimensions must be provided for"
                " dimensionality reduction.")

        self.Ds = int(self.D / self.M)
        self.codebook = []
        self.pqcode = None
        self.membership_ratio = None
        self._fcms = []
        self.inertia = np.empty((self.M))
        
        if add:
            self.pqcode = np.empty((data.shape[0], self.M, 2), self.code_inttype)
            self.membership_ratio = np.empty((data.shape[0], self.M), np.float16)
        
        if self.orth_transf:
            self._O = ortho_group.rvs(dim=self.D)
            data = np.dot(data, self._O)
        
        if self.dim_reduction:
            self._pcas = []

        self._compute_partitions(data, features_labels)
        if num_dims and self.features_cluster_sizes.min() < num_dims:
            raise ValueError("Number of dimensions for dimensionality reduction"
                " must be less than the number of features in each subspace.")
        data = data[:, self._features_perm]

        if add and compute_energy:
            self.energy_std = self.compute_mean_energy(data,
                compute_partitions=False)

        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]

            if num_dims:
                pca = PCA(n_components=num_dims, whiten=whiten).fit(data_sub)
                data_sub_red = pca.transform(data_sub)
                initial_centers = kmeans_plusplus_initializer(data_sub_red,
                    self.K, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
                    ).initialize()
                initial_centers = FDataGrid(initial_centers)
                fcm = FuzzyCMeans(n_clusters=self.K, init=initial_centers,
                    fuzzifier=self.fuzzifier, n_init=1, random_state=self.seed,
                    max_iter=self.kmeans_iter)
                fcm = fcm.fit(FDataGrid(data_sub_red))
                cluster_centers = fcm.cluster_centers_.data_matrix.reshape(-1, pca.n_components_)
                if self.dim_reduction:
                    self._pcas.append(pca)
                    self.codebook.append(cluster_centers)
                else:
                    cluster_centers = pca.inverse_transform(cluster_centers)
                    self.codebook.append(cluster_centers)
            else:
                initial_centers = kmeans_plusplus_initializer(data_sub,
                    self.K, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
                    ).initialize()
                initial_centers = FDataGrid(initial_centers)
                fcm = FuzzyCMeans(n_clusters=self.K, init=initial_centers,
                    fuzzifier=self.fuzzifier, n_init=1, random_state=self.seed,
                    max_iter=self.kmeans_iter)
                fcm = fcm.fit(FDataGrid(data_sub))
                cluster_centers = fcm.cluster_centers_.data_matrix.reshape(-1, data_sub.shape[1])
                self.codebook.append(cluster_centers)

            self.inertia[m] = fcm.inertia_
            self._fcms.append(fcm)

            if verbose:
                print(f"KMeans on subspace {m+1} converged in {fcm.n_iter_}"
                    f" iterations with an inertia of {fcm.inertia_}.")
            
            if add:
                full_membership = fcm.predict_proba(FDataGrid(data_sub))
                sorted_codes = np.argsort(full_membership, axis=1)
                self.pqcode[:, m, 0] = sorted_codes[:, -1]
                self.pqcode[:, m, 1] = sorted_codes[:, -2]
                membership1 = full_membership[range(data_sub.shape[0]), sorted_codes[:, -1]]
                membership2 = full_membership[range(data_sub.shape[0]), sorted_codes[:, -2]]
                self.membership_ratio[:, m] = membership2 / membership1
    
    def add(self, data: np.ndarray, compute_energy: bool = False) -> None:
        """
        Add data to the database.

        Parameters
        ----------

        data : np.ndarray
            Data to add to the database.

        compute_energy : bool, default=False
            Compute the average energy (the sum of squared components) within
            each subspace.

        Notes
        ----

        If called multiple times, the previous data is lost.

        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before adding data.")
        if data.shape[1] != self.D:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        self.pqcode, self.membership_ratio = self.compress(data)

        if self.features_labels is not None:
            data = data[:, self._features_perm]

        if compute_energy:
            self.energy_std = self.compute_mean_energy(data,
                compute_partitions=False)

    def compress(self, data: np.ndarray) -> np.ndarray:
        """
        Compress data using the trained quantizer.

        Parameters
        ----------

        data : np.ndarray
            Data to compress.

        Returns
        -------

        compressed : np.ndarray
            Compressed representation of the data.

        membership : np.ndarray
            Top-2 cluster memberships of the compressed representation.
        
        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before"
                " compressing.")
        if data.shape[1] != self.D:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        if self.orth_transf:
            data = np.dot(data, self._O)

        if self.features_labels is not None:
            data = data[:, self._features_perm]

        codes = np.empty((data.shape[0], self.M, 2), self.code_inttype)
        membership_ratio = np.empty((data.shape[0], self.M), np.float16)
        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            
            if self.dim_reduction:
                data_sub = self._pcas[m].transform(data_sub)
            
            full_membership = self._fcms[m].predict_proba(FDataGrid(data_sub))
            sorted_codes = np.argsort(full_membership, axis=1)
            codes[:, m, 0] = sorted_codes[:, -1]
            codes[:, m, 1] = sorted_codes[:, -2]
            membership1 = full_membership[range(data_sub.shape[0]), sorted_codes[:, -1]]
            membership2 = full_membership[range(data_sub.shape[0]), sorted_codes[:, -2]]
            membership_ratio[:, m] = membership2 / membership1
        
        return codes, membership_ratio
    
    def decompress(self, codes: np.ndarray, membership_ratio: np.ndarray) -> np.ndarray:
        """
        Decompress codes using the trained quantizer.

        Parameters
        ----------

        codes : np.ndarray
            Codes to decompress.

        membership_ratio : np.ndarray
            Membership ratio of the two clusters with highest membership (second
            highest membership / highest membership) for the codes.

        Returns
        -------

        decompressed : np.ndarray
            Decompressed representation of the codes.
        
        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before"
                " decompressing.")
        if codes.shape[1] != self.M:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        decompressed = np.empty((codes.shape[0], self.D), np.float32)
        for m in range(self.M):
            membership = np.empty((codes.shape[0], self.M, 2), np.float16)
            membership[:, m, 0] = 1 / (1 + membership_ratio[:, m])
            membership[:, m, 1] = membership_ratio[:, m] / (1 + membership_ratio[:, m])
            if self.dim_reduction:
                decompressed[:, self._chunk_start[m] : self._chunk_start[m+1]] = \
                    self._pcas[m].inverse_transform(
                        (self.codebook[m][codes[:, m, 0]] * membership[:, m, 0].reshape(-1, 1) + \
                        self.codebook[m][codes[:, m, 1]] * membership[:, m, 1].reshape(-1, 1)))
            else:
                decompressed[:, self._chunk_start[m] : self._chunk_start[m+1]] = \
                    (self.codebook[m][codes[:, m, 0]] * membership[:, m, 0].reshape(-1, 1) + \
                    self.codebook[m][codes[:, m, 1]] * membership[:, m, 1].reshape(-1, 1))
        
        if self.features_labels is not None:
            decompressed = decompressed[:, np.argsort(self._features_perm)]

        if self.orth_transf:
            decompressed = np.dot(decompressed, self._O.T)

        return decompressed
    
    def search(self, query: np.ndarray, subset: np.ndarray = None,
        sort: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the distances of the query to the database vectors.

        Parameters
        ----------

        query : np.ndarray
            Query vector.

        subset : np.ndarray, default=None
            Indices of the database vectors to search in.

        sort : bool, default=True
            Sort the distances returned.

        Returns
        -------

        dists : np.ndarray
            Distances of the query to the database vectors.
        
        idx : np.ndarray
            Indices of the database vectors sorted by distance in increasing
            order, if `sort` is True.
        
        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before searching.")
        if self.pqcode is None:
            raise ValueError("Vectors must be added before searching.")
        if len(query) != self.D:
            raise ValueError("Query dimensions must match trained data"
                " dimensions.")
        if subset is not None and subset.shape[0] > self.pqcode.shape[0]:
            raise ValueError("Subset size must be less or equal to the number"
                " of added vectors.")
        if subset is not None and \
            (subset.max() >= self.pqcode.shape[0] or subset.min() < 0):
            raise ValueError("Subset indices must be non-negative and less than"
                " the number of added vectors.")

        if subset is None:
            n = self.pqcode.shape[0]
            subset = slice(None)
        else:
            n = subset.shape[0]

        if self.orth_transf:
            query = np.dot(query, self._O)

        if self.features_labels is not None:
            query = query[self._features_perm]

        dist_table = np.zeros((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[self._chunk_start[m] : self._chunk_start[m+1]]
            if self.dim_reduction:
                query_sub = self._pcas[m].transform([query_sub]).reshape(-1)
            dist_table[m, :] = cdist([query_sub], self.codebook[m], 'sqeuclidean')[0]
        
        dist = np.zeros(n, dtype=np.float32)
        membership = np.zeros((n, self.M, 2), dtype=np.float16)
        membership_ratio_subset = self.membership_ratio[subset]
        membership[:, :, 0] = 1 / (1 + membership_ratio_subset)
        membership[:, :, 1] = membership_ratio_subset / (1 + membership_ratio_subset)
        for m in range(self.M):
            dist += (
                dist_table[m, self.pqcode[subset, m, 0]] * membership[:, m, 0] +
                dist_table[m, self.pqcode[subset, m, 1]] * membership[:, m, 1]
            )

        if sort:
            return dist, np.argsort(dist)
        return dist, None

class IVF:
    """
    Inverted File (IVF) implementation with Product Quantization (PQ).

    """

    Kp: int
    """
    Number of centroids for the coarse quantizer.
    """
    kmeans_iter: int
    """
    Maximum number of iterations for KMeans.
    """
    kmeans_minit: str
    """
    Method for KMeans initialization.
    """
    seed: int
    """
    Random seed.
    """
    coarse_clust_alg: str
    """
    The clustering algorithm to use for the coarse quantizer.
    """
    batch_size: int
    """
    Batch size for MiniBatchKMeans clustering of the coarse quantizer.
    """
    ivf: list[np.ndarray]
    """
    Inverted index storing data indices assigned to each centroid.
    """
    num_els: int
    """
    Total number of vectors added to the index.
    """
    centroids: np.ndarray
    """
    Coarse quantizer cluster centroids.
    """
    pq: PQ
    """
    Product Quantizer instance for quantizing residuals.
    """
    inertia: float
    """
    Inertia of the KMeans clustering for the coarse quantizer.
    """

    def __init__(self, Kp: int = 1024, M: int = 8, K: int = 256,
        kmeans_iter: int = 300, kmeans_minit: str = "k-means++",
        seed: int = None, orth_transf: bool = False,
        dim_reduction: bool = False, shrink_threshold: float = None,
        coarse_clust_alg: str = "km", batch_size: int = 1024):
        """
        Constructor.

        Parameters
        ----------

        Kp: int, default=1024
            Number of centroids for the coarse quantizer.

        M : int, default=8
            Number of subspaces for the PQ quantizer.
        
        K : int, default=256
            Number of clusters per subspace for the PQ quantizer.
        
        kmeans_iter : int, default=300
            Maximum number of iterations for KMeans.
        
        kmeans_minit : str, default='k-means++'
            Method for KMeans initialization.
            See https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html
        
        seed : int, default None
            Random seed.
        
        orth_transf : bool, default=False
            Apply orthogonal transformation to the data in the PQ quantizer.

        dim_reduction : bool, default=False
            Apply PCA transformation to reduce dimensionality of each subspace
            in the PQ quantizer.

        shrink_threshold : float, default=None
            Threshold for shrinking centroids to remove features.

        coarse_clust_alg : str, default='km'
            The clustering algorithm to use for the coarse quantizer.
            * 'km': KMenas clustering.
            * 'mkm': MiniBatchKMeans clustering.
            * 'bkm': BisectingKMeans clustering.

        batch_size : int, default=1024
            Batch size for MiniBatchKMeans clustering of the coarse quantizer.

        """

        if Kp <= 0:
            raise ValueError("Kp must be greater than 0.")

        self.Kp = Kp
        self.kmeans_iter = kmeans_iter
        self.kmeans_minit = kmeans_minit
        self.seed = seed
        if coarse_clust_alg not in ["km", "bkm", "mkm"]:
            raise ValueError("Supported clustering algorithms for the coarse"
                " quantizer are 'km', 'bkm', or 'mkm'.")
        self.coarse_clust_alg = coarse_clust_alg
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        self.batch_size = batch_size
        
        self.ivf = None
        self.num_els = 0
        self.centroids = None
        self.pq = PQ(M=M, K=K, kmeans_iter=self.kmeans_iter,
            kmeans_minit=self.kmeans_minit, seed=seed, orth_transf=orth_transf,
            dim_reduction=dim_reduction, shrink_threshold=shrink_threshold)
        self.inertia = None
    
    def train(self, data: np.ndarray, add: bool = True,
        compute_distortions: bool = False, compute_energy: bool = False,
        features_labels: np.ndarray = None, num_dims: int = None,
        whiten: bool = False, weight_samples: bool = False, neighbor: int = 3,
        inverse_weights: bool = True, weight_method: str = "normal",
        verbose: bool = False) -> None:
        """
        Train the IVF on the given data.

        Parameters
        ----------

        data : np.ndarray
            Data to train the IVF.

        add : bool, default=True
            Add the data to the index.

        compute_distortions : bool, default=False
            Compute the average distortion for each cluster in each subspace
            (if `add` is also True) for the PQ quantizer.

        compute_energy : bool, default=False
            Compute the average energy (the sum of squared components) within
            each subspace.

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces in the PQ quantizer.

        num_dims : int, default=None
            Number of dimensions in each subspace after PCA dimensionality
            reduction for the PQ quantizer. If `self.dim_reduction` is False,
            but `num_dims` is provided, centroids are computed in the reduced
            space and then transformed back to the original space.

        whiten : bool, default=False
            If True, apply whitening to the PCA transformation.

        weight_samples : bool, default=False
            Weight samples while training KMeans based on the distance to
            the neighbor-th nearest neighbor for the PQ quantizer.

        neighbor : int, default=3
            Neighbor-th nearest neighbor for weighting samples (if
            `weight_samples` is True).

        inverse_weights : bool, default=True
            If True, the weights are inversely proportional to the distance to
            the neighbor-th nearest neighbor (when `weight_samples` is True).

        weight_method : str, default='normal'
            Method for computing weights (when weight_samples is True):
            * 'normal': Normalize the distances to [0, 1].
            * 'reciprocal': If `inverse_weights` is True, compute the reciprocal
                of the distances and normalize to [0, 1], otherwise normalize
                the distances to [0, 1].

        verbose : bool, default=False
            Print training information.
        
        """
        
        if data.shape[0] <= self.Kp:
            raise ValueError("Number of vectors must be greater than the number"
                " of centroids.")
        
        self.num_els = 0
        self.ivf = None

        clust_alg_params = {
            "n_clusters": self.Kp,
            "init": self.kmeans_minit,
            "n_init": 1,
            "random_state": self.seed,
            "max_iter": self.kmeans_iter
        }
        if self.coarse_clust_alg == "km":
            clust_alg = KMeans
        elif self.coarse_clust_alg == "mkm":
            clust_alg = MiniBatchKMeans
            clust_alg_params["batch_size"] = self.batch_size
        else:
            clust_alg = BisectingKMeans
        
        km = clust_alg(**clust_alg_params).fit(data)
        self.inertia = km.inertia_
        
        if verbose:
            print(f"Coarse clustering algorithm converged in {km.n_iter_}"
                " iterations.")

        self.centroids = km.cluster_centers_
        labels, _ = vq(data, self.centroids)

        if add:
            self.ivf = [np.where(labels == i)[0] for i in range(self.Kp)]
            self.num_els = data.shape[0]
        
        residuals = data - self.centroids[labels]
        self.pq.train(data=residuals, add=add,
            compute_distortions=compute_distortions,
            compute_energy=compute_energy, features_labels=features_labels,
            num_dims=num_dims, whiten=whiten, weight_samples=weight_samples,
            neighbor=neighbor, inverse_weights=inverse_weights,
            weight_method=weight_method, verbose=verbose)

    def add(self, data: np.ndarray, compute_distortions:bool = False,
        compute_energy: bool = False) -> None:
        """
        Add data to the IVF structure.

        Parameters
        ----------

        data : np.ndarray
            Data to add to the index.

        compute_distortions : bool, default=False
            Compute the average distortion for each cluster in each subspace
            for the PQ quantizer.

        compute_energy : bool, default=False
            Compute the average energy (the sum of squared components) within
            each subspace.
        
        """
        
        if self.centroids is None:
            raise ValueError("The index must be created before adding data.")
        if data.shape[1] != self.pq.D:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        labels, _ = vq(data, self.centroids)
        self.ivf = [np.where(labels == i)[0] for i in range(self.Kp)]
        self.num_els = data.shape[0]
        
        residuals = data - self.centroids[labels]
        self.pq.add(data=residuals, compute_distortions=compute_distortions,
            compute_energy=compute_energy)

    def search(self, query: np.ndarray, w: int = 8, asym: bool = True,
        correct: bool = False, sort: bool = True) \
        -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the distances of the query to the database vectors.

        Parameters
        ----------

        query : np.ndarray
            Query vector.

        w : int, default=8
            Number of coarse centroids to visit.

        asym : bool, default=True
            Use asymmetric distance computation (do not quantize the query).

        correct : bool, default=False
            Correct distances by adding average distortions.

        sort : bool, default=True
            Sort the distances returned.

        Returns
        -------

        dists : np.ndarray
            Distances of the query to the database vectors, sorted in increasing
            order if `sort` is True.
        
        idx : np.ndarray
            Indices of the database vectors sorted by distance in increasing
            order, if `sort` is True.
        
        """

        if w > self.Kp:
            raise ValueError("Number of centroids to visit must be less or equal"
                " to the number of centroids.")
        if self.centroids is None:
            raise ValueError("The index must be trained before searching.")
        if self.ivf is None:
            raise ValueError("Vectors must be added before searching.")
        if len(query) != self.pq.D:
            raise ValueError("Query dimensions must match trained data"
                " dimensions.")

        dist2centroids = cdist([query], self.centroids, 'sqeuclidean')[0]
        sorted_centroids = np.argsort(dist2centroids)
        els_per_centroid = np.array(
            [len(self.ivf[centroid]) for centroid in sorted_centroids])
        num_els = np.sum(els_per_centroid[ : w])
        dists = np.empty(num_els, np.float32)
        els = np.empty(num_els, np.int64)
        
        for i in range(w):
            query_res = query - self.centroids[sorted_centroids[i]]
            curr_items = self.ivf[sorted_centroids[i]]
            if curr_items.shape[0] == 0:
                continue
            curr_dist, _ = self.pq.search(query_res, subset=curr_items,
                asym=asym, correct=correct, sort=False)
            num_prev_items = np.sum(els_per_centroid[ : i])
            num_curr_items = els_per_centroid[i]
            dists[num_prev_items : num_prev_items + num_curr_items] = curr_dist
            els[num_prev_items : num_prev_items + num_curr_items] = curr_items

        if sort:
            sorted_idx = np.argsort(dists)
            dists = dists[sorted_idx]
            els = els[sorted_idx]
        
        return dists, els
    
class FuzzyIVF:
    """
    Inverted File (IVF) implementation with Fuzzy Product Quantization (PQ).

    """

    Kp: int
    """
    Number of centroids for the coarse quantizer.
    """
    kmeans_iter: int
    """
    Maximum number of iterations for KMeans.
    """
    kmeans_minit: str
    """
    Method for KMeans initialization.
    """
    seed: int
    """
    Random seed.
    """
    coarse_clust_alg: str
    """
    The clustering algorithm to use for the coarse quantizer.
    """
    batch_size: int
    """
    Batch size for MiniBatchKMeans clustering of the coarse quantizer.
    """
    ivf: list[np.ndarray]
    """
    Inverted index storing data indices assigned to each centroid.
    """
    num_els: int
    """
    Total number of vectors added to the index.
    """
    centroids: np.ndarray
    """
    Coarse quantizer cluster centroids.
    """
    pq: PQ
    """
    Product Quantizer instance for quantizing residuals.
    """
    inertia: float
    """
    Inertia of the KMeans clustering for the coarse quantizer.
    """

    def __init__(self, Kp: int = 1024, M: int = 8, K: int = 256,
        kmeans_iter: int = 300, kmeans_minit: str = "k-means++",
        fuzzifier: float = 2, seed: int = None, orth_transf: bool = False,
        dim_reduction: bool = False, coarse_clust_alg: str = "km",
        batch_size: int = 1024):
        """
        Constructor.

        Parameters
        ----------

        Kp: int, default=1024
            Number of centroids for the coarse quantizer.

        M : int, default=8
            Number of subspaces for the PQ quantizer.
        
        K : int, default=256
            Number of clusters per subspace for the PQ quantizer.
        
        kmeans_iter : int, default=300
            Maximum number of iterations for KMeans.
        
        kmeans_minit : str, default='k-means++'
            Method for KMeans initialization.
            See https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html

        fuzzifier : float, default=2
            Hyper-parameter that controls how fuzzy the cluster will be.
            The higher it is, the fuzzier the cluster will be in the end.
            This parameter should be greater than 1.
        
        seed : int, default None
            Random seed.
        
        orth_transf : bool, default=False
            Apply orthogonal transformation to the data in the PQ quantizer.

        dim_reduction : bool, default=False
            Apply PCA transformation to reduce dimensionality of each subspace
            in the PQ quantizer.

        coarse_clust_alg : str, default='km'
            The clustering algorithm to use for the coarse quantizer.
            * 'km': KMenas clustering.
            * 'mkm': MiniBatchKMeans clustering.
            * 'bkm': BisectingKMeans clustering.

        batch_size : int, default=1024
            Batch size for MiniBatchKMeans clustering of the coarse quantizer.

        """

        if Kp <= 0:
            raise ValueError("Kp must be greater than 0.")

        self.Kp = Kp
        self.kmeans_iter = kmeans_iter
        self.kmeans_minit = kmeans_minit
        self.seed = seed
        if coarse_clust_alg not in ["km", "bkm", "mkm"]:
            raise ValueError("Supported clustering algorithms for the coarse"
                " quantizer are 'km', 'bkm', or 'mkm'.")
        self.coarse_clust_alg = coarse_clust_alg
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        self.batch_size = batch_size
        
        self.ivf = None
        self.num_els = 0
        self.centroids = None
        self.pq = FuzzyPQ(M=M, K=K, kmeans_iter=self.kmeans_iter,
            fuzzifier=fuzzifier, seed=seed, orth_transf=orth_transf,
            dim_reduction=dim_reduction)
        self.inertia = None
    
    def train(self, data: np.ndarray, add: bool = True,
        compute_energy: bool = False, features_labels: np.ndarray = None,
        num_dims: int = None, whiten: bool = False, verbose: bool = False) -> None:
        """
        Train the IVF on the given data.

        Parameters
        ----------

        data : np.ndarray
            Data to train the IVF.

        add : bool, default=True
            Add the data to the index.

        compute_energy : bool, default=False
            Compute the average energy (the sum of squared components) within
            each subspace.

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces in the PQ quantizer.

        num_dims : int, default=None
            Number of dimensions in each subspace after PCA dimensionality
            reduction for the PQ quantizer. If `self.dim_reduction` is False,
            but `num_dims` is provided, centroids are computed in the reduced
            space and then transformed back to the original space.

        whiten : bool, default=False
            If True, apply whitening to the PCA transformation.

        verbose : bool, default=False
            Print training information.
        
        """
        
        if data.shape[0] <= self.Kp:
            raise ValueError("Number of vectors must be greater than the number"
                " of centroids.")
        
        self.num_els = 0
        self.ivf = None

        clust_alg_params = {
            "n_clusters": self.Kp,
            "init": self.kmeans_minit,
            "n_init": 1,
            "random_state": self.seed,
            "max_iter": self.kmeans_iter
        }
        if self.coarse_clust_alg == "km":
            clust_alg = KMeans
        elif self.coarse_clust_alg == "mkm":
            clust_alg = MiniBatchKMeans
            clust_alg_params["batch_size"] = self.batch_size
        else:
            clust_alg = BisectingKMeans
        
        km = clust_alg(**clust_alg_params).fit(data)
        self.inertia = km.inertia_
        
        if verbose:
            print(f"Coarse clustering algorithm converged in {km.n_iter_}"
                " iterations.")

        self.centroids = km.cluster_centers_
        labels, _ = vq(data, self.centroids)

        if add:
            self.ivf = [np.where(labels == i)[0] for i in range(self.Kp)]
            self.num_els = data.shape[0]
        
        residuals = data - self.centroids[labels]
        self.pq.train(data=residuals, add=add, compute_energy=compute_energy,
            features_labels=features_labels, num_dims=num_dims, whiten=whiten,
            verbose=verbose)

    def add(self, data: np.ndarray, compute_energy: bool = False) -> None:
        """
        Add data to the IVF structure.

        Parameters
        ----------

        data : np.ndarray
            Data to add to the index.

        compute_energy : bool, default=False
            Compute the average energy (the sum of squared components) within
            each subspace.
        
        """
        
        if self.centroids is None:
            raise ValueError("The index must be created before adding data.")
        if data.shape[1] != self.pq.D:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        labels, _ = vq(data, self.centroids)
        self.ivf = [np.where(labels == i)[0] for i in range(self.Kp)]
        self.num_els = data.shape[0]
        
        residuals = data - self.centroids[labels]
        self.pq.add(data=residuals, compute_energy=compute_energy)

    def search(self, query: np.ndarray, w: int = 8, sort: bool = True) \
        -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the distances of the query to the database vectors.

        Parameters
        ----------

        query : np.ndarray
            Query vector.

        w : int, default=8
            Number of coarse centroids to visit.

        sort : bool, default=True
            Sort the distances returned.

        Returns
        -------

        dists : np.ndarray
            Distances of the query to the database vectors, sorted in increasing
            order if `sort` is True.
        
        idx : np.ndarray
            Indices of the database vectors sorted by distance in increasing
            order, if `sort` is True.
        
        """

        if w > self.Kp:
            raise ValueError("Number of centroids to visit must be less or equal"
                " to the number of centroids.")
        if self.centroids is None:
            raise ValueError("The index must be trained before searching.")
        if self.ivf is None:
            raise ValueError("Vectors must be added before searching.")
        if len(query) != self.pq.D:
            raise ValueError("Query dimensions must match trained data"
                " dimensions.")

        dist2centroids = cdist([query], self.centroids, 'sqeuclidean')[0]
        sorted_centroids = np.argsort(dist2centroids)
        els_per_centroid = np.array(
            [len(self.ivf[centroid]) for centroid in sorted_centroids])
        num_els = np.sum(els_per_centroid[ : w])
        dists = np.empty(num_els, np.float32)
        els = np.empty(num_els, np.int64)
        
        for i in range(w):
            query_res = query - self.centroids[sorted_centroids[i]]
            curr_items = self.ivf[sorted_centroids[i]]
            if curr_items.shape[0] == 0:
                continue
            curr_dist, _ = self.pq.search(query_res, subset=curr_items,
                sort=False)
            num_prev_items = np.sum(els_per_centroid[ : i])
            num_curr_items = els_per_centroid[i]
            dists[num_prev_items : num_prev_items + num_curr_items] = curr_dist
            els[num_prev_items : num_prev_items + num_curr_items] = curr_items

        if sort:
            sorted_idx = np.argsort(dists)
            dists = dists[sorted_idx]
            els = els[sorted_idx]
        
        return dists, els
    
class ExactSearch:
    """
    Exact search implementation.

    """

    data: np.ndarray
    """
    The dataset in which to search.
    """

    def __init__(self, data: np.ndarray):
        """
        Constructor.

        Parameters
        ----------

        data : np.ndarray
            The dataset in which to search.
        
        """
        self.data = data

    def search(self, query: np.ndarray, sort: bool = True) \
        -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the distances of the query to the dataset vectors.

        Parameters
        ----------

        query : np.ndarray
            Query vector.

        sort : bool, default=True
            Sort the distances returned.

        Returns
        -------

        dists : np.ndarray
            Distances of the query to the dataset vectors.

        idx : np.ndarray
            Indices of the dataset vectors sorted by distance in increasing
            order, if `sort` is True.
        
        """
        
        if len(query) != self.data.shape[1]:
            raise ValueError("Query dimensions must match dataset dimensions.")

        dist = np.sum((self.data - query)**2, axis=1)   

        if sort:
            return dist, np.argsort(dist)
        
        return dist, None