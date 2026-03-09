import numpy as np
from sklearn.metrics import silhouette_score
from .kmeans import fit_kmeans

def pick_k_by_silhouette(X, k_min=2, k_max=8, random_state=0):

    # Tests k in [k_min..k_max], returns:  best_k, scores_dict
    scores = {}
    best_k = None
    best_score = -np.inf

    for k in range(k_min, k_max + 1):
        labels, _, _, Xs = fit_kmeans(X, k=k, random_state=random_state)
        # silhouette needs at least 2 clusters and not all points same label
        if len(set(labels)) < 2:
            scores[k] = float("nan")
            continue
        s = silhouette_score(Xs, labels)
        scores[k] = float(s)
        if s > best_score:
            best_score = s
            best_k = k

    return best_k, scores
