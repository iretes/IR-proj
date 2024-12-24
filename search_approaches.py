import numpy as np
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans, BisectingKMeans, SpectralBiclustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PQ:
    def __init__(self, M: int = 8, K: int = 256, kmeans_iter: int = 300,
        kmeans_minit:str = "k-means++", seed:int = None,
        orth_transf: bool = False, part_opt_alg: str = None,
        PCA_transf: bool = False):
        """
        Product Quantization (PQ) implementation.

        Attributes:
            M (int): Number of subspaces.
            K (int): Number of clusters per subspace.
            kmeans_iter (int): Maximum number of iterations for KMeans.
            kmeans_minit (str): Method for KMeans initialization.
            seed (int): Random seed.
            code_inttype (numpy.dtype): Integer type for storing codes.
            codebook (numpy.ndarray): Cluster centroids for each subspace.
            Ds (int): Dimension of each subspace.
            D (int): Original feature dimension.
            pqcode (numpy.ndarray): Quantized representation of the data.
            avg_dist (numpy.ndarray): Average distortion for each cluster in each subspace.
            inertia (numpy.ndarray): Inertia of the KMeans clustering in each subspace.
            part_opt_alg (str): Algorithm for optimizing partitions.
            PCA_transf (bool): Apply PCA transformation to the data.
            orth_transf (bool): Apply orthogonal transformation to the data.
            cols_perm (numpy.ndarray): Permutation of the columns.
            chunk_start (numpy.ndarray): Start index of each subspace in the data.
            col_cluster_sizes (numpy.ndarray): Number of vectors in each cluster.
        """
         # TODO: checks

        self.M = M
        self.K = K
        self.kmeans_iter = kmeans_iter
        self.kmeans_minit = kmeans_minit
        self.seed = seed

        K_bits = np.log2(self.K-1)
        if K_bits <= 8:
            self.code_inttype = np.uint8
        elif K_bits <= 16:
            self.code_inttype = np.uint16
        elif K_bits <= 32:
            self.code_inttype = np.uint32
        else:
            self.code_inttype = np.uint64

        self.codebook = None
        self.Ds = None
        self.pqcode = None
        self.avg_dist = None
        self.inertia = None
        self.orth_transf = orth_transf
        self.part_opt_alg = part_opt_alg
        self.PCA_transf = PCA_transf
        self.chunk_start = None
        self.col_cluster_sizes = None

    def _optimize_partitions(self, data: np.ndarray,
        col_labels: np.ndarray = None) -> None:
        """Optimize the partitions of the data based on the KMeans clustering of
        the columns."""
        
        if self.part_opt_alg:
            if self.part_opt_alg == "custom":
                self.col_labels = col_labels
            elif self.part_opt_alg == "sbc":
                sbc = SpectralBiclustering(n_clusters=(self.K, self.M),
                    n_init=1, random_state=self.seed) # TODO: altri param?
                sbc.fit(data)
                self.col_labels = sbc.column_labels_
            elif self.part_opt_alg == "km":
                km = KMeans(n_clusters=self.M, init=self.kmeans_minit,
                    n_init=1, random_state=self.seed, max_iter=self.kmeans_iter)
                km = km.fit(data.T)
                self.col_labels = km.labels_
            _, self.col_cluster_sizes = np.unique(self.col_labels, return_counts=True)
            self.chunk_start = np.zeros(self.M+1, dtype=int)
            self.chunk_start[1:] = np.cumsum(self.col_cluster_sizes)
            self.cols_perm = np.argsort(self.col_labels)
        else:
            self.chunk_start = np.arange(0, self.M * self.Ds + self.Ds, self.Ds)
            self.col_cluster_sizes = np.full(self.M, self.Ds)
            self.cols_perm = np.arange(data.shape[1])  # identity permutation

    def plot_neighbor_distances(self, data: np.ndarray, neighbor: int,
        ax: plt.Axes) -> None:
        """Plot the distances to the neighbor-th nearest neighbor for each
        vector in the dataset in each subspace."""
        
        self.D = data.shape[1]
        assert self.D % self.M == 0, "Feature dimension must be divisible by the number of subspaces (M)."
        self.Ds = int(self.D / self.M)

        self._optimize_partitions(data, None)
        data = data[:, self.cols_perm]

        for m in range(self.M):
            data_sub = data[:, self.chunk_start[m] : self.chunk_start[m+1]]
            knn = NearestNeighbors(n_neighbors=neighbor).fit(data_sub)
            distances, _ = knn.kneighbors(data_sub)
            ax.plot(np.sort(distances[:, -1]), label=f"Subspace {m+1}")
        ax.set_ylabel(f"{neighbor}-th nearest neighbor distance")
        ax.set_xlabel("Vectors")
        ax.legend()
    
    def plot_variance_explained(self, data: np.ndarray) -> None:
        """Plot the variance explained by each principal component for each
        subspace."""
        
        self.D = data.shape[1]
        assert self.D % self.M == 0, "Feature dimension must be divisible by the number of subspaces (M)."
        self.Ds = int(self.D / self.M)

        self._optimize_partitions(data, None)
        data = data[:, self.cols_perm]

        for m in range(self.M):
            data_sub = data[:, self.chunk_start[m] : self.chunk_start[m+1]]
            pca = PCA().fit(data_sub)
            plt.plot(pca.explained_variance_ratio_, label=f"Subspace {m+1}")
        plt.ylabel("Variance explained")
        plt.xlabel("Principal components")
        plt.legend()

    def _neighbor_distances_to_weights(self, distances: np.ndarray,
        inverse_weights: bool, weight_method: str) -> np.ndarray:
        """Convert distances to weights for KMeans clustering."""

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
        """Compute weights for KMeans clustering based on the distance to the
        neighbor-th nearest neighbor."""
        
        knn = NearestNeighbors(n_neighbors=neighbor).fit(data)
        distances, _ = knn.kneighbors(data)
        return self._neighbor_distances_to_weights(distances[:, -1], inverse_weights, weight_method)

    def train(self, data: np.ndarray, add:bool = True,
        compute_distortions:bool = False, weight_samples:bool = False,
        neighbor:int = 3, inverse_weights: bool = True,
        weight_method: str = "normal", num_dims: int = None,
        col_labels: np.ndarray = None, verbose:bool = False) -> None:
        """ Train the quantizer on the given data."""

        # TODO: assert sui nuovi parametri
        self.D = data.shape[1]
        assert self.D % self.M == 0, "Feature dimension must be divisible by the number of subspaces (M)."
        self.Ds = int(self.D / self.M)
        self.codebook = []
        self.inertia = np.empty((self.M))
        self.pqcode = None # if train is called twice, previous codes are discarded
        self.avg_dist = None
        if self.part_opt_alg == "custom":
            assert col_labels is not None, "Column labels must be provided for partition optimization."
        
        if add:
            self.pqcode = np.empty((data.shape[0], self.M), self.code_inttype)
            if compute_distortions:
                self.avg_dist = np.zeros((self.M, self.K), np.float32)

        if self.PCA_transf:
            self.pca = PCA().fit(data)
            data = self.pca.transform(data)

        if self.orth_transf:
            rng = np.random.default_rng(self.seed)
            A = rng.random((self.D, self.D))
            self.Q, _ = np.linalg.qr(A)
            data = np.dot(data, self.Q)

        self._optimize_partitions(data, col_labels)
        data = data[:, self.cols_perm]

        for m in range(self.M):
            data_sub = data[:, self.chunk_start[m] : self.chunk_start[m+1]]
            sample_weight = None
            if weight_samples:
                sample_weight = self._compute_clustering_weights(data_sub, neighbor, inverse_weights, weight_method)
            km = KMeans(n_clusters=self.K, init=self.kmeans_minit, n_init=1,
                        random_state=self.seed, max_iter=self.kmeans_iter)
            if num_dims:
                pca = PCA().fit(data_sub)
                data_sub_red = pca.transform(data_sub)
                data_sub_red = data_sub_red[:, :num_dims]
                km = km.fit(data_sub_red, sample_weight=sample_weight)
                cluster_centers = np.matmul(km.cluster_centers_, pca.components_[ : num_dims, : ]) + pca.mean_
                self.codebook.append(cluster_centers)
            else:
                km = km.fit(data_sub, sample_weight=sample_weight)
                self.codebook.append(km.cluster_centers_)
            self.inertia[m] = km.inertia_
            
            if verbose:
                print(f"KMeans on subspace {m+1} converged in {km.n_iter_} iterations with an inertia of {km.inertia_}.")
            
            if add:
                self.pqcode[:, m], _ = vq(data_sub, self.codebook[m])
                if compute_distortions:
                    for k in range(self.K):
                        dist = cdist(data_sub[self.pqcode[:, m] == k], [self.codebook[m][k]], 'sqeuclidean')
                        self.avg_dist[m, k] = np.mean(dist)

    # NOTE: una sola volta
    def add(self, data: np.ndarray, compute_distortions:bool = False) -> None:
        """ Add data to the quantizer."""

        assert self.codebook is not None, "The quantizer must be trained before adding data."
        assert data.shape[1] == self.D, "Data dimensions must match trained data dimensions."

        pqcode = self.compress(data)
        self.pqcode = pqcode # if self.pqcode is None else np.vstack((self.pqcode, pqcode))

        if compute_distortions: # recomputed if we train on subspace and add other data
            if self.part_opt_alg:
                data = data[:, self.cols_perm]
            self.avg_dist = np.zeros((self.M, self.K), np.float32)
            for m in range(self.M):
                data_sub = data[:, self.chunk_start[m] : self.chunk_start[m+1]]
                for k in range(self.K):
                    dist = cdist(data_sub[self.pqcode[:, m] == k], [self.codebook[m][k]], 'sqeuclidean')
                    self.avg_dist[m, k] = np.mean(dist)

    def compress(self, data: np.ndarray) -> np.ndarray:
        """ Compress data using the trained quantizer."""

        assert self.codebook is not None, "The quantizer must be trained before compressing."
        assert data.shape[1] == self.D, "Data dimensions must match trained data dimensions."

        if self.PCA_transf:
            data = self.pca.transform(data)

        if self.orth_transf:
            data = np.dot(data, self.Q)

        if self.part_opt_alg:
            data = data[:, self.cols_perm] # NOTE: permuta

        compressed = np.empty((data.shape[0], self.M), self.code_inttype)
        for m in range(self.M):
            data_sub = data[:, self.chunk_start[m] : self.chunk_start[m+1]]
            compressed[:, m], _ = vq(data_sub, self.codebook[m])
        return compressed
    
    def decompress(self, codes: np.ndarray) -> np.ndarray:
        """ Decompress codes using the trained quantizer."""

        assert self.codebook is not None, "The quantizer must be trained before decompressing."
        assert codes.shape[1] == self.M, "Data dimensions must match trained data dimensions."

        decompressed = np.empty((codes.shape[0], self.D), np.float32)
        for m in range(self.M):
            decompressed[:, self.chunk_start[m] : self.chunk_start[m+1]] = self.codebook[m][codes[:, m]]
        return decompressed

    def search(self, query: np.ndarray, subset: np.ndarray = None,
        asym:bool = True, correct:bool = False,
        sort:bool = True) -> tuple[np.ndarray, np.ndarray]:
        """ Search for the nearest neighbors of the query in the quantized data."""

        assert self.codebook is not None, "The quantizer must be trained before searching."
        assert self.pqcode is not None, "Vectors must be added before searching."
        assert (not correct) or (self.avg_dist is not None), "Distorsion must be computed before correcting."
        assert len(query) == self.D, "Query dimensions must match trained data dimensions."
        assert (subset is None) or (subset.shape[0] <= self.pqcode.shape[0]), "Subset size must be less or equal to the number of added vectors."

        if subset is None:
            subset = slice(None)

        if self.PCA_transf:
            query = self.pca.transform(query.reshape(1, -1)).reshape(-1)

        if self.orth_transf:
            query = np.dot(query, self.Q)

        if self.part_opt_alg:
            query = query[self.cols_perm]

        dist_table = np.empty((self.M, self.K), np.float32)
        for m in range(self.M):
            query_sub = query[self.chunk_start[m] : self.chunk_start[m+1]]
            if not asym:
                query_sub_code, _ = vq([query_sub], self.codebook[m])
                query_sub = self.codebook[m][query_sub_code[0]]
            dist_table[m, :] = cdist([query_sub], self.codebook[m], 'sqeuclidean')[0]
            if correct:
                dist_table[m, :] += self.avg_dist[m]
                if not asym:
                    dist_table[m, query_sub_code] += self.avg_dist[m, query_sub_code]

        dist = np.sum(dist_table[range(self.M), self.pqcode[subset]], axis=1)
        
        if sort:
            return dist, np.argsort(dist)
        return dist, None
    
class IVF:
    def __init__(self, Kp: int = 1024, M:int = 8, K:int = 256,
        kmeans_iter:int = 300, kmeans_minit:str = "k-means++",
        seed:int = None, part_opt_alg:str = None, bisectingkmeans: bool = False):
        """
        Inverted File (IVF) implementation with Product Quantization (PQ).
    
        Attributes:
            Kp (int): Number of clusters for the coarse quantizer.
            kmeans_iter (int): Maximum number of iterations for KMeans.
            kmeans_minit (str): Method for KMeans initialization.
            seed (int, optional): Random seed.
            ivf (list of np.ndarray): Inverted index storing data indices assigned to each centroid.
            num_els (int): Total number of vectors added to the index.
            centroids (np.ndarray): Coarse quantizer cluster centroids.
            pq (PQ): Product Quantizer instance for quantizing residuals.
        """

        self.Kp = Kp
        self.kmeans_iter = kmeans_iter
        self.kmeans_minit = kmeans_minit
        self.seed = seed
        self.bisectingkmeans = bisectingkmeans
        self.ivf = None
        self.num_els = 0
        self.centroids = None
        self.inertia = None
        self.pq = PQ(M=M, K=K, kmeans_iter=self.kmeans_iter,
                     kmeans_minit=self.kmeans_minit, seed=None, part_opt_alg=part_opt_alg)

    def train(self, data: np.ndarray, add:bool = True,
        compute_distortions:bool = False, weight_samples:bool = False,
        neighbor:int = 3, inverse_weights: bool = True,
        weight_method: str = "normal", num_dims: int = None,
        col_labels: np.ndarray = None, verbose:bool = False) -> None:
        """Train the IVF on the given data."""
        
        assert data.shape[0] > self.Kp, "Number of vectors must be greater than the number of centroids."
        
        self.num_els = 0
        self.ivf = None

        clustering_algorithm = BisectingKMeans if self.bisectingkmeans else KMeans
        km = clustering_algorithm(n_clusters=self.Kp, init=self.kmeans_minit, n_init=1,
            random_state=self.seed, max_iter=self.kmeans_iter).fit(data)
        self.inertia = km.inertia_
        if verbose:
            print(f"KMeans for IVF converged in {km.n_iter_} iterations.")
        
        self.centroids = km.cluster_centers_
        labels, _ = vq(data, self.centroids)

        if add:
            self.ivf = [np.where(labels == i)[0] for i in range(self.Kp)]
            self.num_els = data.shape[0]
        
        residuals = data - self.centroids[labels]
        self.pq.train(residuals, add=add,
            compute_distortions=compute_distortions,
            weight_samples=weight_samples, neighbor=neighbor,
            inverse_weights=inverse_weights, weight_method=weight_method,
            num_dims=num_dims, col_labels=col_labels, verbose=verbose)

    # NOTE: una sola volta
    def add(self, data: np.ndarray, compute_distortions:bool = False) -> None:
        """Add data to the IVF structure."""
        
        assert self.centroids is not None, "The index must be created before adding data."
        assert data.shape[1] == self.pq.D, "Data dimensions must match trained data dimensions."

        labels, _ = vq(data, self.centroids)
        self.ivf = [np.where(labels == i)[0] for i in range(self.Kp)]
        self.num_els = data.shape[0]
        
        residuals = data - self.centroids[labels]
        self.pq.add(residuals, compute_distortions=compute_distortions)

        # for i in range(self.Kp):
        #     els_i = np.where(labels == i)[0]
        #     if els_i.shape[0] > 0:
        #         self.ivf[i] = np.hstack((self.ivf[i], els_i))

    def search(self, query: np.ndarray, w:int = 8, asym:bool = True,
        correct:bool = False, sort:bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Search for the closest vectors to the query in the IVF index."""
        
        assert w <= self.Kp, "Number of centroids to visit must be less or equal to the number of centroids."
        assert self.centroids is not None, "The index must be created before searching."
        assert self.ivf is not None, "Vectors must be added before searching."
        assert len(query) == self.pq.D, "Query dimensions must match trained data dimensions."

        dist2centroids = cdist([query], self.centroids, 'sqeuclidean')[0]
        sorted_centroids = np.argsort(dist2centroids)
        els_per_centroid = np.array([len(self.ivf[centroid]) for centroid in sorted_centroids])
        num_els = np.sum(els_per_centroid[:w])
        dists = np.empty(num_els, np.float32)
        els = np.empty(num_els, np.int64)
        
        for i in range(w):
            query_res = query - self.centroids[sorted_centroids[i]]
            curr_docs = self.ivf[sorted_centroids[i]]
            if curr_docs.shape[0] == 0:
                continue
            curr_dist, _ = self.pq.search(query_res, subset=curr_docs, asym=asym, correct=correct, sort=False)
            num_prev_docs = np.sum(els_per_centroid[:i])
            num_curr_docs = els_per_centroid[i]
            dists[num_prev_docs:num_prev_docs+num_curr_docs] = curr_dist
            els[num_prev_docs:num_prev_docs+num_curr_docs] = curr_docs

        if sort:
            sorted_idx = np.argsort(dists)
            dists = dists[sorted_idx] # NOTE: different from PQ where distances are not sorted
            els = els[sorted_idx]
        
        return dists, els
    
class ExactSearch:
    def __init__(self, data: np.ndarray):
        """
        Implements exact search.
        
        Attributes:
            data (np.ndarray): The dataset in which to search.
        """
        
        self.data = data

    def search(self, query: np.ndarray,
        sort:bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Searches for the closest vectors to the query."""
        
        assert len(query) == self.data.shape[1], "Query dimensions must match dataset dimensions."

        dist = np.sum((self.data - query)**2, axis=1)   

        if sort:
            return dist, np.argsort(dist)
        
        return dist, None