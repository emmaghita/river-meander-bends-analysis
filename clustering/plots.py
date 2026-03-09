import matplotlib.pyplot as plt
import numpy as np

def plot_feature_scatter(X, labels, feature_names, title="KMeans clustering (feature space)"):

    # Plots pairwise scatter plots for 3 features.

    if X.shape[1] != 3:
        raise ValueError("This plotting function expects exactly 3 features.")

    f1, f2, f3 = feature_names

    # 1) f1 vs f2
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=18, alpha=0.9)
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_title(f"{title}: {f1} vs {f2}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2) f1 vs f3
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    ax.scatter(X[:, 0], X[:, 2], c=labels, s=18, alpha=0.9)
    ax.set_xlabel(f1)
    ax.set_ylabel(f3)
    ax.set_title(f"{title}: {f1} vs {f3}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3) f2 vs f3
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    ax.scatter(X[:, 1], X[:, 2], c=labels, s=18, alpha=0.9)
    ax.set_xlabel(f2)
    ax.set_ylabel(f3)
    ax.set_title(f"{title}: {f2} vs {f3}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cluster_centers(model, feature_names, title="Cluster centers (scaled space)"):
    centers = model.cluster_centers_
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=150)
    for i, c in enumerate(centers):
        ax.plot(np.arange(len(c)), c, marker="o", label=f"cluster {i}")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
