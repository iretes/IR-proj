import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from search_approaches import PQ

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
    Quantized representation of the data added to the database (codes of the two
    nearest centroids).
    """
    weights: np.ndarray
    """
    Weights of the codes corresponding to the nearest centroids for the data
    added data in the database.
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

        super().__init__(M, K, kmeans_iter, kmeans_minit, seed, orth_transf,
            part_alg, dim_reduction)
        self.weights = None
    
    def _compute_nearest_codes(self, data: np.ndarray,
        codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the distances to two closest codes for each vector in the data.

        Parameters
        ----------

        data : np.ndarray
            Data to compute the distances to the closest codes.

        codebook : np.ndarray
            Codebook.

        Returns
        -------

        sorted_codes : np.ndarray
            Codes of the two nearest centroids, sorted by distance.
        
        sorted_distances : np.ndarray
            Distances to the two nearest centroids, sorted.

        """
        
        distances = cdist(data, codebook)
        sorted_codes = np.argsort(distances, axis=1)
        sorted_distances = np.take_along_axis(distances, sorted_codes, axis=1)
        return sorted_codes[:, :2], sorted_distances[:, :2]

    def train(self, data: np.ndarray, add: bool = True,
        compute_energy: bool = False, features_labels: np.ndarray = None,
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
        self.weights = None
        self.inertia = np.empty((self.M))
        
        if add:
            self.pqcode = np.empty((data.shape[0], self.M, 2), self.code_inttype)
            self.weights = np.empty((data.shape[0], self.M), np.float16)
        
        if self.orth_transf:
            from scipy.stats import ortho_group
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
                sorted_codes, sorted_dists = self._compute_nearest_codes(data_sub, self.codebook[m])
                self.pqcode[:, m, 0] = sorted_codes[:, 0] # closest code
                self.pqcode[:, m, 1] = sorted_codes[:, 1] # second closest code
                self.weights[:, m] = sorted_dists[:, 1] / (sorted_dists[:, 0] + sorted_dists[:, 1]) # weight for the closest code

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

        self.pqcode, self.weights = self.compress(data)

        if self.part_alg:
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

        weight : np.ndarray
            Weights of the compressed representation.
        
        """

        if self.codebook is None:
            raise ValueError("The quantizer must be trained before"
                " compressing.")
        if data.shape[1] != self.D:
            raise ValueError("Data dimensions must match trained data"
                " dimensions.")

        if self.orth_transf:
            data = np.dot(data, self._O)

        if self.part_alg:
            data = data[:, self._features_perm]

        codes = np.empty((data.shape[0], self.M, 2), self.code_inttype)
        weights = np.empty((data.shape[0], self.M), np.float16)
        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            
            if self.dim_reduction:
                data_sub = self._pcas[m].transform(data_sub)
            
            sorted_codes, sorted_dists = self._compute_nearest_codes(data_sub, self.codebook[m])
            codes[:, m, 0] = sorted_codes[:, 0]
            codes[:, m, 1] = sorted_codes[:, 1]
            weights[:, m] = sorted_dists[:, 1] / (sorted_dists[:, 0] + sorted_dists[:, 1])
        
        return codes, weights
    
    def decompress(self, codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Decompress codes using the trained quantizer.

        Parameters
        ----------

        codes : np.ndarray
            Codes to decompress.

        weights : np.ndarray
            Weights of the compressed representation.

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
                    self._pcas[m].inverse_transform(self.codebook[m][codes[:, m, 0]] * weights[:, m].reshape(-1, 1) + \
                        self.codebook[m][codes[:, m, 1]] * (1 - weights[:, m].reshape(-1, 1)))
            else:
                decompressed[:, self._chunk_start[m] : self._chunk_start[m+1]] = \
                    self.codebook[m][codes[:, m, 0]] * weights[:, m].reshape(-1, 1) + \
                    self.codebook[m][codes[:, m, 1]] * (1 - weights[:, m].reshape(-1, 1))
        
        if self.part_alg:
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
            num_vectors_to_search = self.pqcode.shape[0]
            subset = slice(None)
        else:
            num_vectors_to_search = subset.shape[0]

        if self.orth_transf:
            query = np.dot(query, self._O)

        if self.part_alg:
            query = query[self._features_perm]

        # dist = np.zeros(num_vectors_to_search, dtype=np.float32)
        # for m in range(self.M):
        #     average_centroid = (
        #         (self.codebook[m][self.pqcode[subset, m][:, 0]] * self.pqcodeweights[subset, m][:, np.newaxis] +
        #         self.codebook[m][self.pqcode[subset, m][:, 1]] * (1 - self.pqcodeweights[subset, m][:, np.newaxis])))

        #     query_sub = query[self._chunk_start[m]:self._chunk_start[m + 1]]
        #     if self.dim_reduction:
        #         query_sub = self._pcas[m].transform([query_sub]).reshape(-1)
        #     dist += np.sum((query_sub - average_centroid) ** 2, axis=1)

        # Precompute query distances to centroids for each subspace
        query_dists = np.zeros((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[self._chunk_start[m] : self._chunk_start[m+1]]
            if self.dim_reduction:
                query_sub = self._pcas[m].transform([query_sub]).reshape(-1)
            query_dists[m, :] = cdist([query_sub], self.codebook[m], 'sqeuclidean')[0]

        # Compute distances using weighted centroid combinations
        dist = np.zeros(num_vectors_to_search, dtype=np.float32)
        for m in range(self.M):
            codes_closest = self.pqcode[subset, m, 0]
            codes_second = self.pqcode[subset, m, 1]
            weights_closest = self.weights[subset, m]

            dist_closest = query_dists[m, codes_closest]
            dist_second = query_dists[m, codes_second]

            dist += dist_closest * weights_closest + dist_second * (1 - weights_closest)
        
        if sort:
            return dist, np.argsort(dist)
        return dist, None