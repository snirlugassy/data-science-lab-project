import pickle
import numpy as np
import pandas as pd
from pysparnn.cluster_index import ClusterIndex
from scipy import sparse


def append_related_to(nn_index_path, vectors_path, data_path, output_path):
    with open(nn_index_path, 'rb') as f:
        index = pickle.load(f)
    type(index)

    vectors = pickle.load(open(vectors_path,'rb'))
    data = pd.read_csv(data_path)

    data['related_to'] = 0

    # assign to each company the nearest neighbor using the index to 'related_to'
    # each company is also considered as neighbor for itself,
    # therefore, consider only the second neighbor
    for i in range(data.shape[0]):
        print(i, end='\r')
        data.at[i, 'related_to'] = index.search(vectors[i], k=2, return_distance=False)[0][1]

    data['related_industry'] = data.related_to.apply(lambda i: data.iloc[i].industry)
    data.to_csv(output_path, index=False)

if __name__ == '__main__':
    append_related_to('nn_index.pkl', 'vectors.pkl', 'data.csv', 'data.csv')