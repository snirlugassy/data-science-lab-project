import pickle
from sklearn.cluster import KMeans, MiniBatchKMeans


def kmeans_clustering(vectors):
    with open('vectorizer.pkl', 'rb') as f:
        X = pickle.load(f)

    kmeans = KMeans(n_clusters=10, random_state=0, max_iter=10, verbose=True)
    labels = kmeans.fit_predict(vectors)

    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    with open('doc2cluster.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
    return kmeans, labels

if __name__ == '__main__':
    with open('vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)
    
    kmeans, labels = kmeans_clustering(vectors)