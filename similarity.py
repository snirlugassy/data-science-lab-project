import pickle
from sklearn.cluster import KMeans

seed = 42

def kmeans_clustering(K, vectors):
    print('Initializing KMeans')
    kmeans = KMeans(n_clusters=K, random_state=seed, verbose=True)
    
    print('Fitting vectors')
    kmeans.fit(vectors)
    
    print('Predicting clusters')
    labels = kmeans.predict(vectors)

    print('Saving KMeans fitted model to kmeans_model.pkl')
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    print('Saving assigned clusters for each document to doc2cluster.pkl')
    with open('doc2cluster.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
    return kmeans, labels

if __name__ == '__main__':
    print('Loading vectors from vectors.pkl')
    with open('vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)
        print('Loaded vectors with shape', vectors.shape)
    
    K = 6
    print('Starting KMeans clustering using K=', K)
    kmeans, labels = kmeans_clustering(K, vectors)
    print('Done')