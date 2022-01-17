import pickle
import pandas as pd
from pysparnn.cluster_index import MultiClusterIndex

vectors = pickle.load(open('vectors.pkl','rb'))
data = pd.read_csv('data.csv')

index = MultiClusterIndex(vectors, data)

with open('nn_index.pkl', 'wb') as f:
    pickle.dump(index, f)