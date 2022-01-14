import sys
import threading
import pickle
import pandas as pd
from scipy import sparse


def cosine(x:sparse.spmatrix,y: sparse.spmatrix):
    return (x * y.T).sum() / ((x * x.T) * (y * y.T)).sqrt().sum()


class ClusterPairwiseSimilarityThread(threading.Thread):
    def __init__(self, df, cluster_id, vectors, output_file) -> None:
        threading.Thread.__init__(self)
        assert 'cluster' in df.columns
        assert 'industry' in df.columns
        self.df = df
        self.vectors = vectors
        self.cluster_id = cluster_id
        self.output_file = output_file
    
    def __get_pairs(self):
        _df = self.df[self.df.cluster == self.cluster_id]
        return (
            (i1, i2) 
            for i1, x1 in _df.iterrows() 
            for i2, x2 in _df.iterrows() 
            if i1 < i2 and x1.industry != x2.industry
        )

    def run(self):
        print('Calculating pairwise cosine similarity for cluster ' + str(self.cluster_id))
        with open(self.output_file, 'w') as out:
            for i,j in self.__get_pairs():
                out.write(f'{i},{j},{cosine(self.vectors[i], self.vectors[j])}\n')
        print('Finished cluster ' + str(self.cluster_id))

if __name__ == '__main__':
    input_file = sys.argv[1]
    data = pd.read_csv(input_file)

    with open('vectors.pkl', 'rb') as f:
        vectors = pickle.load(f)

    threads = []
    for c in data.cluster.unique():
        t = ClusterPairwiseSimilarityThread(data, c, vectors, f'cluster_{c}_similarities.txt')
        t.start()
        threads.append(t)
