import pickle
from sklearn.cluster import KMeans


def kmeans_clustering(vectors):
    print('1')
    kmeans = KMeans(n_clusters=6, random_state=0, max_iter=10, verbose=True)
    print('2')
    labels = kmeans.fit_predict(vectors)
    print('3')

    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    with open('doc2cluster.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
    return kmeans, labels

if __name__ == '__main__':
    with open('vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)
    
    kmeans, labels = kmeans_clustering(vectors)