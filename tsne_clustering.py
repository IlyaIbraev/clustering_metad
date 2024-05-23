from multiprocessing import Pool
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sklearn.utils.validation as suv


from tsne_manager import TSNEManager


class TSNEKMeans:

    perplexity_values = range(100, 1100, 100)
    clusters_values = [
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100
    ]

    def __init__(self, features_filename: str, num_proc: int, manager: TSNEManager) -> None:
        self.features_filename = features_filename
        self.num_proc = num_proc
        self.manager = manager

    def _read_rmsd_matrix(self) -> None:
        data = np.fromfile(self.features_filename, dtype=np.float32)
        self.rmsd = data.reshape(
            int(np.sqrt(len(data))), int(np.sqrt(len(data)))
        )

    def _check_symmetry(self) -> None:
        assert suv.check_symmetric(
            self.rmsd, raise_exception=True), "Matrix is not symmetric"

    def _calculate_tsne(self, perplexity: int) -> None:

        tsne_model = TSNE(
            n_components=2,
            perplexity=perplexity,
            early_exaggeration=10.0,
            learning_rate=100.0,
            n_iter=3500,
            n_iter_without_progress=300,
            min_grad_norm=1e-7,
            metric="precomputed",
            init="random",
            method="barnes_hut",
            angle=0.5
        )

        tsne_transformed = tsne_model.fit_transform(self.rmsd)

        np.savetxt(
            f"calculations/{self.features_filename}/transformed/{perplexity}",
            tsne_transformed
        )

    def _proceed_calculations(self) -> None:

        with Pool(self.num_proc) as pool:
            pool.map(self._calculate_tsne, self.perplexity_values)

    def _proceed_kmeans(self) -> None:

        for perplexity in self.perplexity_values:
            # fix later
            tsne_transformed = np.loadtxt(
                f"calculations/{self.features_filename}/transformed/{perplexity}.dat"
            )
            for n_clusters in self.clusters_values:
                kmeans_model = KMeans(
                    n_clusters=n_clusters,
                    n_init="auto",
                )
                kmeans_model.fit(tsne_transformed)
                cluster_labels = kmeans_model.labels_
                np.savetxt(
                    f"calculations/{self.features_filename}/clusters/{perplexity}/{n_clusters}.dat",
                    cluster_labels
                )

                low_dim_score = silhouette_score(
                    tsne_transformed, cluster_labels, metric="euclidean"
                )
                high_dim_score = silhouette_score(
                    self.rmsd, cluster_labels, metric="precomputed"
                )
                self.manager.log(
                    self.features_filename, perplexity, n_clusters, low_dim_score*high_dim_score
                )

    def proceed_clustering(self) -> None:

        self._read_rmsd_matrix()
        self._check_symmetry()
        self._proceed_calculations()
        self._proceed_kmeans()
