import pickle
import pandas as pd
from pysparnn.cluster_index import MultiClusterIndex, ClusterIndex

def build_index(vectors_path, output_path):
    vectors = pickle.load(open(vectors_path,'rb'))

    index = ClusterIndex(vectors, list(range(vectors.shape[0])))

    with open(output_path, 'wb') as f:
        pickle.dump(index, f)

if __name__ == '__main__':
    build_index('vectors.pkl', 'nn_index.pkl')