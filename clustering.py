import pickle
import sys
import gc
import pandas as pd
from sklearn.cluster import KMeans

seed = 42

def kmeans_clustering(K, vectors):
    print('Initializing KMeans')
    kmeans = KMeans(n_clusters=K, n_init=1, random_state=seed, verbose=True, tol=1e-2, init='random')
    
    print('Fitting clusters')
    kmeans.fit(vectors)
    
    print('Predicting clusters')
    labels = kmeans.predict(vectors)

    print('Saving KMeans fitted model to kmeans_model.pkl')
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    return kmeans, labels

if __name__ == '__main__':
    print('Loading vectors from vectors.pkl')
    with open('vectors.pkl', 'rb') as f:
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

    if len(sys.argv) > 1:
        data_file_path = sys.argv[1]
    else:
        data_file_path = input('Please enter the text csv file path:')
    
    print('Loading text dataframe')
    data = pd.read_csv(data_file_path)

    print('Assignin cluster to documents')
    data['cluster'] = labels

    print('Saving')
    data.to_csv(data_file_path, index=False)

    print('Done')