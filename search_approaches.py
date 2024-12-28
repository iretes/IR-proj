import numpy as np
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans, BisectingKMeans, SpectralBiclustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    part_alg: str
    """
    Algorithm for partitioning the features into subspaces.
    """
    dim_reduction: bool
    """
    Apply PCA transformation to reduce dimensionality of each subspace.
    """
    Ds: int
    """
    Dimension of each subspace, when `self.part_alg` is None (equal
    partitioning).
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

    def __init__(self, M: int = 8, K: int = 256, kmeans_iter: int = 300,
        kmeans_minit: str = "k-means++", seed: int = None,
        orth_transf: bool = False, part_alg: str = None,
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
            Maximum number of iterations for KMeans.
        
        kmeans_minit : str, default='k-means++'
            Method for KMeans initialization.
            See https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html
        
        seed : int, default None
            Random seed.
        
        orth_transf : bool, default=False
            Apply orthogonal transformation to the data.
        
        part_alg : str, default=None
            Algorithm for partitioning the features into subspaces.
            * None: Equally partition the features into subspaces.
            * 'custom': Custom partitioning, column labels must be provided to
                the train method.
            * 'sbc': Spectral Biclustering of the data.
            * 'km': KMeans clustering of the columns.

        dim_reduction : bool, default=False
            Apply PCA transformation to reduce dimensionality of each subspace.

        """

        if M <= 0:
            raise ValueError("M must be greater than 0.")
        if K <= 0:
            raise ValueError("K must be greater than 0.")
        if part_alg not in [None, "custom", "sbc", "km"]:
            raise ValueError("Supported partitioning algorithms are 'custom',"
                " 'sbc', or 'km'.")

        self.M = M
        self.K = K
        self.kmeans_iter = kmeans_iter
        self.kmeans_minit = kmeans_minit
        self.seed = seed
        self.orth_transf = orth_transf
        self.part_alg = part_alg
        self.dim_reduction = dim_reduction

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
        self._Q = None

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
        
        if self.part_alg:
            if self.part_alg == "custom":
                self.features_labels = features_labels
            elif self.part_alg == "sbc":
                sbc = SpectralBiclustering(n_clusters=(self.K, self.M),
                    n_init=1, random_state=self.seed)
                sbc.fit(data)
                self.features_labels = sbc.column_labels_
            elif self.part_alg == "km":
                km = KMeans(n_clusters=self.M, init=self.kmeans_minit,
                    n_init=1, random_state=self.seed, max_iter=self.kmeans_iter)
                km = km.fit(data.T)
                self.features_labels = km.labels_
            _, self.features_cluster_sizes = np.unique(self.features_labels,
                return_counts=True)
            self._chunk_start = np.zeros(self.M+1, dtype=int)
            self._chunk_start[1:] = np.cumsum(self.features_cluster_sizes)
            self._features_perm = np.argsort(self.features_labels)
        else:
            self.features_cluster_sizes = np.full(self.M, self.Ds)
            self._chunk_start = np.arange(0, self.M * self.Ds + self.Ds, self.Ds)
            self._features_perm = np.arange(data.shape[1])  # identity permutation

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
        compute_distortions: bool = False, features_labels: np.ndarray = None,
        num_dims: int = None, weight_samples: bool = False, neighbor: int = 3,
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

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces.

        num_dims : int, default=None
            Number of dimensions in each subspace after PCA dimensionality
            reduction. If `self.dim_reduction` is False, but `num_dims` is
            provided, centroids are computed in the reduced space and then
            transformed back to the original space.

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
        if self.part_alg == "custom" and features_labels is None:
            raise ValueError("Feature labels must be provided for custom"
                " partitioning.")
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

        self.Ds = int(self.D / self.M)
        self.codebook = []
        self.pqcode = None
        self.avg_dist = None
        self.inertia = np.empty((self.M))
        
        if add:
            self.pqcode = np.empty((data.shape[0], self.M), self.code_inttype)
            if compute_distortions:
                self.avg_dist = np.zeros((self.M, self.K), np.float32)
        
        if self.orth_transf:
            rng = np.random.default_rng(self.seed)
            A = rng.random((self.D, self.D))
            self._Q, _ = np.linalg.qr(A)
            data = np.dot(data, self._Q)
        
        if self.dim_reduction:
            self._pcas = []

        self._compute_partitions(data, features_labels)
        if num_dims and self.features_cluster_sizes.min() < num_dims:
            raise ValueError("Number of dimensions for dimensionality reduction"
                " must be less than the number of features in each subspace.")
        data = data[:, self._features_perm]

        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            
            sample_weight = None
            if weight_samples:
                sample_weight = self._compute_clustering_weights(data_sub,
                    neighbor, inverse_weights, weight_method)
            
            km = KMeans(n_clusters=self.K, init=self.kmeans_minit, n_init=1,
                random_state=self.seed, max_iter=self.kmeans_iter)
            
            if num_dims:
                pca = PCA(n_components=num_dims).fit(data_sub)
                data_sub_red = pca.transform(data_sub)
                km = km.fit(data_sub_red, sample_weight=sample_weight)
                if self.dim_reduction:
                    self._pcas.append(pca)
                    self.codebook.append(km.cluster_centers_)
                else:
                    cluster_centers = pca.inverse_transform(km.cluster_centers_)
                    self.codebook.append(cluster_centers)
            else:
                km = km.fit(data_sub, sample_weight=sample_weight)
                self.codebook.append(km.cluster_centers_)

            self.inertia[m] = km.inertia_
            
            if verbose:
                print(f"KMeans on subspace {m+1} converged in {km.n_iter_}"
                    f" iterations with an inertia of {km.inertia_}.")
            
            if add:
                self.pqcode[:, m], _ = vq(data_sub, self.codebook[m])
                if compute_distortions:
                    for k in range(self.K):
                        dist = cdist(data_sub[self.pqcode[:, m] == k],
                            [self.codebook[m][k]], 'sqeuclidean')
                        self.avg_dist[m, k] = np.mean(dist)

    def add(self, data: np.ndarray, compute_distortions: bool = False) -> None:
        """
        Add data to the database.

        Parameters
        ----------

        data : np.ndarray
            Data to add to the database.
        
        compute_distortions : bool, default=False
            Compute the average distortion for each cluster in each subspace.

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

        if compute_distortions:
            if self.part_alg:
                data = data[:, self._features_perm]
            
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
            data = np.dot(data, self._Q)

        if self.part_alg:
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
        
        if self.part_alg:
            decompressed = decompressed[:, np.argsort(self._features_perm)]

        if self.orth_transf:
            decompressed = np.dot(decompressed, self._Q.T)
        
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
            query = np.dot(query, self._Q)

        if self.part_alg:
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
    bisectingkmeans: bool
    """
    Use BisectingKMeans instead of KMeans for the coarse quantizer.
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
        seed: int = None, orth_transf: bool = False, part_alg: str = None,
        dim_reduction: bool = False, bisectingkmeans: bool = False):
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
        
        part_alg : str, default=None
            Algorithm for partitioning the features into subspaces in the PQ
            quantizer:
            * None: Equally partition the features into subspaces.
            * 'custom': Custom partitioning, column labels must be provided to
                the train method.
            * 'sbc': Spectral Biclustering of the data.
            * 'km': KMeans clustering of the columns.

        dim_reduction : bool, default=False
            Apply PCA transformation to reduce dimensionality of each subspace
            in the PQ quantizer.

        bisectingkmeans : bool, default=False
            Use BisectingKMeans instead of KMeans for the coarse quantizer.

        """

        if Kp <= 0:
            raise ValueError("Kp must be greater than 0.")

        self.Kp = Kp
        self.kmeans_iter = kmeans_iter
        self.kmeans_minit = kmeans_minit
        self.seed = seed
        self.bisectingkmeans = bisectingkmeans
        
        self.ivf = None
        self.num_els = 0
        self.centroids = None
        self.pq = PQ(M=M, K=K, kmeans_iter=self.kmeans_iter,
            kmeans_minit=self.kmeans_minit, seed=seed, orth_transf=orth_transf,
            part_alg=part_alg, dim_reduction=dim_reduction)
        self.inertia = None

    def train(self, data: np.ndarray, add: bool = True,
        compute_distortions: bool = False, weight_samples: bool = False,
        neighbor: int = 3, inverse_weights: bool = True,
        weight_method: str = "normal", num_dims: int = None,
        features_labels: np.ndarray = None, verbose: bool = False) -> None:
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
        
        num_dims : int, default=None
            Number of dimensions in each subspace after PCA dimensionality
            reduction for the PQ quantizer. If `self.dim_reduction` is False,
            but `num_dims` is provided, centroids are computed in the reduced
            space and then transformed back to the original space.

        features_labels : np.ndarray, default=None
            Features labels for custom partitioning of the features into
            subspaces in the PQ quantizer.

        verbose : bool, default=False
            Print training information.
        
        """
        
        if data.shape[0] <= self.Kp:
            raise ValueError("Number of vectors must be greater than the number"
                " of centroids.")
        
        self.num_els = 0
        self.ivf = None

        clust_alg = BisectingKMeans if self.bisectingkmeans else KMeans
        km = clust_alg(n_clusters=self.Kp, init=self.kmeans_minit,
            n_init=1, random_state=self.seed,
            max_iter=self.kmeans_iter).fit(data)
        self.inertia = km.inertia_
        
        if verbose:
            print(f"KMeans for IVF converged in {km.n_iter_} iterations.")
        
        self.centroids = km.cluster_centers_
        labels, _ = vq(data, self.centroids)

        if add:
            self.ivf = [np.where(labels == i)[0] for i in range(self.Kp)]
            self.num_els = data.shape[0]
        
        residuals = data - self.centroids[labels]
        self.pq.train(data=residuals, add=add,
            compute_distortions=compute_distortions,
            features_labels=features_labels, num_dims=num_dims,
            weight_samples=weight_samples, neighbor=neighbor,
            inverse_weights=inverse_weights, weight_method=weight_method,
            verbose=verbose)

    def add(self, data: np.ndarray, compute_distortions:bool = False) -> None:
        """
        Add data to the IVF structure.

        Parameters
        ----------

        data : np.ndarray
            Data to add to the index.

        compute_distortions : bool, default=False
            Compute the average distortion for each cluster in each subspace
            for the PQ quantizer.
        
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
        self.pq.add(data=residuals, compute_distortions=compute_distortions)

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
            curr_docs = self.ivf[sorted_centroids[i]]
            if curr_docs.shape[0] == 0:
                continue
            curr_dist, _ = self.pq.search(query_res, subset=curr_docs,
                asym=asym, correct=correct, sort=False)
            num_prev_docs = np.sum(els_per_centroid[ : i])
            num_curr_docs = els_per_centroid[i]
            dists[num_prev_docs : num_prev_docs + num_curr_docs] = curr_dist
            els[num_prev_docs : num_prev_docs + num_curr_docs] = curr_docs

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