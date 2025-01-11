import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.fcm import fcm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from search_approaches import PQ
from scipy.stats import ortho_group

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
    clusters with highest membership).
    """
    membership: np.ndarray
    """
    Top-2 cluster memberships of the data added to the database.
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
        m: float = 2, seed: int = None, orth_transf: bool = False,
        part_alg: str = None, dim_reduction: bool = False):
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

        m : float, default=2
            Hyper-parameter that controls how fuzzy the cluster will be.
            The higher it is, the fuzzier the cluster will be in the end.
            This parameter should be greater than 1.
        
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

        super().__init__(M=M, K=K, kmeans_iter=kmeans_iter, seed=seed,
            orth_transf=orth_transf, part_alg=part_alg,
            dim_reduction=dim_reduction)
        self.m = m
        self.membership = None
    
    # def _compute_nearest_codes(self, data: np.ndarray,
    #     codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Compute the distances to two closest codes for each vector in the data.

    #     Parameters
    #     ----------

    #     data : np.ndarray
    #         Data to compute the distances to the closest codes.

    #     codebook : np.ndarray
    #         Codebook.

    #     Returns
    #     -------

    #     sorted_codes : np.ndarray
    #         Codes of the two nearest centroids, sorted by distance.
        
    #     sorted_distances : np.ndarray
    #         Distances to the two nearest centroids, sorted.

    #     """
        
    #     distances = cdist(data, codebook)
    #     sorted_codes = np.argsort(distances, axis=1)
    #     sorted_distances = np.take_along_axis(distances, sorted_codes, axis=1)
    #     return sorted_codes[:, :2], sorted_distances[:, :2]
    
    def _compute_membership(self, data, centers): # TODO: add doc
        membership = np.zeros((len(data), len(centers)))

        data_difference = np.zeros((len(centers), len(data)))
        for i in range(len(centers)):
            data_difference[i] = np.sum(np.square(data - centers[i]), axis=1)

        for i in range(len(data)):
            for j in range(len(centers)):
                divider = sum([pow(data_difference[j][i] / data_difference[k][i], self.m) for k in range(len(centers)) if data_difference[k][i] != 0.0])
                if divider != 0.0:
                    membership[i][j] = 1.0 / divider
                else:
                    membership[i][j] = 1.0

        return membership

    def train(self, data: np.ndarray, add: bool = True,
        compute_energy: bool = False, features_labels: np.ndarray = None,
        num_dims: int = None, verbose: bool = False) -> None:
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

        self.Ds = int(self.D / self.M)
        self.codebook = []
        self.pqcode = None
        self.membership = None
        
        if add:
            self.pqcode = np.empty((data.shape[0], self.M, 2), self.code_inttype)
            self.membership = np.empty((data.shape[0], self.M, 2), np.float16)
        
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
                pca = PCA(n_components=num_dims).fit(data_sub)
                data_sub_red = pca.transform(data_sub)
                initial_centers = kmeans_plusplus_initializer(data_sub_red,
                    self.K, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
                    ).initialize()
                fcm_inst = fcm(data_sub_red, initial_centers, m=self.m,
                    itermax=self.kmeans_iter, ccore=False)
                fcm_inst.process()
                if self.dim_reduction:
                    self._pcas.append(pca)
                    self.codebook.append(fcm_inst.get_centers())
                else:
                    cluster_centers = pca.inverse_transform(fcm_inst.get_centers())
                    self.codebook.append(cluster_centers)
            else:
                initial_centers = kmeans_plusplus_initializer(data_sub,
                    self.K, kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
                    ).initialize()
                fcm_inst = fcm(data_sub, initial_centers, m=self.m,
                    itermax=self.kmeans_iter, ccore=False)
                fcm_inst.process()
                self.codebook.append(fcm_inst.get_centers())

            if verbose:
                print(f"Subspace {m+1} trained.")
            
            if add:
                full_membership = self._compute_membership(data_sub, self.codebook[m])
                sorted_codes = np.argsort(full_membership, axis=1)
                self.pqcode[:, m, 0] = sorted_codes[:, -1]
                self.pqcode[:, m, 1] = sorted_codes[:, -2]
                self.membership[:, m, 0] = full_membership[range(full_membership.shape[0]), sorted_codes[:, -1]]
                self.membership[:, m, 1] = full_membership[range(full_membership.shape[0]), sorted_codes[:, -2]]

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

        self.pqcode, self.membership = self.compress(data)

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

        if self.part_alg:
            data = data[:, self._features_perm]

        codes = np.empty((data.shape[0], self.M, 2), self.code_inttype)
        membership = np.empty((data.shape[0], self.M, 2), np.float16)
        for m in range(self.M):
            data_sub = data[:, self._chunk_start[m] : self._chunk_start[m+1]]
            
            if self.dim_reduction:
                data_sub = self._pcas[m].transform(data_sub)
            
            full_membership = self._compute_membership(data_sub, self.codebook[m])
            sorted_codes = np.argsort(full_membership, axis=1)
            codes[:, m, 0] = sorted_codes[:, -1]
            codes[:, m, 1] = sorted_codes[:, -2]
            membership[:, m, 0] = full_membership[range(membership.shape[0]), sorted_codes[:, -1]]
            membership[:, m, 1] = full_membership[range(membership.shape[0]), sorted_codes[:, -2]]
        
        return codes, membership
    
    def decompress(self, codes: np.ndarray, membership: np.ndarray) -> np.ndarray:
        """
        Decompress codes using the trained quantizer.

        Parameters
        ----------

        codes : np.ndarray
            Codes to decompress.

        membership : np.ndarray
            Top-2 cluster memberships of the compressed representation.

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
                    self._pcas[m].inverse_transform(
                        (self.codebook[m][codes[:, m, 0]] * membership[:, m, 0].reshape(-1, 1) + \
                        self.codebook[m][codes[:, m, 1]] * membership[:, m, 1].reshape(-1, 1)) / \
                        ((membership[:, m, 0] + membership[:, m, 1]).reshape(-1, 1)))
            else:
                decompressed[:, self._chunk_start[m] : self._chunk_start[m+1]] = \
                    (self.codebook[m][codes[:, m, 0]] * membership[:, m, 0].reshape(-1, 1) + \
                    self.codebook[m][codes[:, m, 1]] * membership[:, m, 1].reshape(-1, 1)) / \
                    ((membership[:, m, 0] + membership[:, m, 1]).reshape(-1, 1))
                # decompressed[:, self._chunk_start[m] : self._chunk_start[m + 1]] = \
                #     np.sum(self.codebook[m][codes[:, m]] * membership[:, m].reshape(-1, 2, 1), axis=1) / \
                #     membership[:, m].sum(axis=1, keepdims=True)
        
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

        dist_table = np.zeros((self.M, self.K), dtype=np.float32)
        for m in range(self.M):
            query_sub = query[self._chunk_start[m] : self._chunk_start[m+1]]
            if self.dim_reduction:
                query_sub = self._pcas[m].transform([query_sub]).reshape(-1)
            dist_table[m, :] = cdist([query_sub], self.codebook[m], 'sqeuclidean')[0]

        dist = np.zeros(num_vectors_to_search, dtype=np.float32)
        # for m in range(self.M):
        #     dist += (dist_table[m, self.pqcode[subset, m, 0]] * self.membership[subset, m, 0] + \
        #         dist_table[m, self.pqcode[subset, m, 1]] * self.membership[subset, m, 1]) / \
        #         (self.membership[subset, m, 0] + self.membership[subset, m, 1]) 
        dist = np.sum(
            (dist_table[range(self.M), self.pqcode[subset, :, 0]] * self.membership[subset, :, 0] +
            dist_table[range(self.M), self.pqcode[subset, :, 1]] * self.membership[subset, :, 1]) /
            (self.membership[subset, :, 0] + self.membership[subset, :, 1]), axis=1)
        
        if sort:
            return dist, np.argsort(dist)
        return dist, None