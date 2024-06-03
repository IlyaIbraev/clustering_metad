from typing import Dict, Tuple

import numpy as np
import pandas as pd


class TSNEManager:
    def __init__(self, log_filename: str) -> None:
        self.log_filename = log_filename

    def init_file(self) -> None:
        data = pd.DataFrame(
            columns=["feature", "perplexity", "clusters", "silhouette_score"]
        )

        data.to_csv(self.log_filename, index=False)

    def log(
        self, feature: str, perplexity: int, clusters: int, silhouette_score: float
    ) -> None:
        data = pd.read_csv(self.log_filename, index_col=None)

        new_row = [
            {
                "feature": feature,
                "perplexity": perplexity,
                "clusters": clusters,
                "silhouette_score": silhouette_score,
            }
        ]

        data = pd.concat([data, pd.DataFrame(new_row)], ignore_index=True)

        data.to_csv(self.log_filename, index=False)

    def get_best(self) -> Dict[str, pd.DataFrame]:

        data = pd.read_csv(self.log_filename)

        best = {}
        for feature in data.feature.unique():
            feature_data = data[data.feature == feature]
            best[feature] = feature_data[
                feature_data.silhouette_score == feature_data.silhouette_score.max()
            ]

        return best

    def get_all(self) -> pd.DataFrame:
        return pd.read_csv(self.log_filename)

    def get_transformed_and_labels(
        self, method: str, perplexity: int, clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        transformed = np.loadtxt(f"calculations/{method}/transformed/{perplexity}.dat")
        labels = np.loadtxt(
            f"calculations/{method}/clusters/{perplexity}/{clusters}.dat"
        )
        return transformed, labels
