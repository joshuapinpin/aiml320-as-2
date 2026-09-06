import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


class DBSCANEvaluator:
    """Small-grid search over (eps, min_points) for DBSCAN.

    For each combination, records the number of clusters found (excluding
    noise), the fraction of points labelled noise, and a silhouette score
    computed on the non-noise points (when at least 2 clusters exist).
    """

    def __init__(self, X, eps_values=(0.3, 0.5, 0.7, 1.0, 1.5), min_points_values=(3, 5, 8, 10)):
        self.X = X
        self.eps_values = eps_values
        self.min_points_values = min_points_values
        self.results = []

    def evaluate(self):
        """Fit DBSCAN for every (eps, min_points) pair and record the metrics."""
        self.results = []
        for eps in self.eps_values:
            for min_points in self.min_points_values:
                labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(self.X)
                n_clusters = len(set(labels) - {-1})
                noise_frac = float(np.mean(labels == -1))

                sil = np.nan
                if n_clusters >= 2:
                    mask = labels != -1
                    if len(set(labels[mask])) >= 2:
                        sil = silhouette_score(self.X[mask], labels[mask])

                self.results.append({
                    "eps": eps,
                    "min_points": min_points,
                    "n_clusters": n_clusters,
                    "noise_frac": noise_frac,
                    "silhouette": sil,
                })
        return self.table()

    def table(self):
        """Return results as a DataFrame, sorted by eps then min_points."""
        return pd.DataFrame(self.results).sort_values(["eps", "min_points"]).reset_index(drop=True)

    def filter_by_clusters(self, n_clusters):
        """Return rows matching an exact target cluster count, lowest noise first."""
        df = self.table()
        return df[df["n_clusters"] == n_clusters].sort_values("noise_frac").reset_index(drop=True)