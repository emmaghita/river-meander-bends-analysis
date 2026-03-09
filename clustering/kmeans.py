from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def fit_kmeans(X, k=4, random_state=0, n_init=50):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X_scaled)

    return labels, model, scaler, X_scaled
