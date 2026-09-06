import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansEvaluator:
    """Class to evaluate K-means clustering using WCSS and silhouette scores.

    Evaluates K-means for different values of K. The code will compute the WCSS (inertia) and
    silhouette scores for K in the range 2 to 10, and plot the results.
    It will also print the best K based on the silhouette score.
    """
    def __init__(self, X, k_min=2, k_max=10, random_state=42):
        self.X = X
        self.k_range = range(k_min, k_max + 1)
        self.random_state = random_state
        self.wcss = []
        self.silhouette_scores = []

    def evaluate_wcss(self):
        """Fit K-means for each K and record inertia (WCSS)."""
        self.wcss = []
        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto").fit(self.X)
            self.wcss.append(km.inertia_)
        return self.wcss

    def evaluate_silhouette(self):
        """Fit K-means for each K and record the silhouette score."""
        self.silhouette_scores = []
        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto").fit(self.X)
            self.silhouette_scores.append(silhouette_score(self.X, km.labels_))
        return self.silhouette_scores

    def best_k(self):
        """Return the K with the highest silhouette score."""
        best_index = int(np.argmax(self.silhouette_scores))
        return list(self.k_range)[best_index]

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(list(self.k_range), self.wcss, marker="o")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("WCSS (inertia)")
        axes[0].set_title("Elbow Method")
        axes[0].grid(True)

        axes[1].plot(list(self.k_range), self.silhouette_scores, marker="o")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Score vs K")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def main(self):
        print("Evaluating K-means clustering...")
        self.evaluate_wcss()
        self.evaluate_silhouette()
        print("Plotting WCSS and Silhouette scores...")
        self.plot()
        print(f"Best K by silhouette score: {self.best_k()} (score={max(self.silhouette_scores):.4f})")

if __name__ == "__main__":
    print("Loading data from part1_data.npz...")
    X = np.load("part1_data.npz")["X"]
    evaluator = KMeansEvaluator(X)
    evaluator.main()