import pickle
import sys
import gc
import pandas as pd
from sklearn.cluster import KMeans
from vectorizer import VECTORS_FILE

seed = 42
MODEL_FILE = 'kmeans_model.pkl'

def kmeans_clustering(K, vectors):
    print('Initializing KMeans')
    kmeans = KMeans(n_clusters=K, n_init=1, random_state=seed, verbose=True, tol=1e-2, init='random')
    
    print('Fitting clusters')
    kmeans.fit(vectors)
    
    print('Predicting clusters')
    labels = kmeans.predict(vectors)

    print(f'Saving KMeans fitted model to {MODEL_FILE}')
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(kmeans, f)
    
    return kmeans, labels

if __name__ == '__main__':
    data_file_path = sys.argv[1]

    print(f'Loading vectors from {VECTORS_FILE}')
    with open(VECTORS_FILE, 'rb') as f:
        vectors = pickle.load(f)
        print('Loaded vectors with shape', vectors.shape)
    
    K = 6
    print('Starting KMeans clustering using K=', K)
    kmeans, labels = kmeans_clustering(K, vectors)

    print('Cleaning memory')
    # Release memory
    del vectors
    del kmeans

    # Garbage collection
    gc.collect()

    print('Loading text dataframe')
    data = pd.read_csv(data_file_path)

    print('Assignin cluster to documents')
    data['cluster'] = labels

    print('Saving')
    data.to_csv(data_file_path, index=False)

    print('Done')