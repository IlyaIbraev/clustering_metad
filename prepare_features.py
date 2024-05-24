from typing import Callable, List

from tsne_clustering import TSNEKMeans
from tsne_manager import TSNEManager
from prepare_filestructure import prepare_feature_calc_directory


class Feature:
    def __init__(self, name: str, prepare_function: Callable, params: dict) -> None:
        self.name = name
        self.prepare_function = prepare_function
        self.params = params


class FeaturesManager:

    features: List[Feature] = []

    def __init__(
        self,
        # num_proc: int,
        manager: TSNEManager
    ) -> None:
        # self.num_proc = num_proc
        self.manager = manager

    def add_feature(self, feature: Feature) -> None:
        self.features.append(feature)

    def prepare_features(self) -> None:
        for feature in self.features:
            prepare_feature_calc_directory(feature.name)
            feature.prepare_function(**feature.params)

    def proceed_clustering(self) -> None:
        for feature in self.features:
            tsne_kmeans = TSNEKMeans(
                feature.name,
                # self.num_proc,
                self.manager
            )
            tsne_kmeans.proceed_clustering()
