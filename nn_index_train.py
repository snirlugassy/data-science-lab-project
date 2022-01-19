import pickle
import pandas as pd
from pysparnn.cluster_index import MultiClusterIndex, ClusterIndex

vectors = pickle.load(open('vectors.pkl','rb'))

index = ClusterIndex(vectors, list(range(vectors.shape[0])))

with open('nn_index.pkl', 'wb') as f:
    pickle.dump(index, f)