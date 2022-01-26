import os
import sys
import subprocess
from typing import Iterable

from utils.merge_csv import merge
from preprocessing import preprocessing
from vectorizer import train_vectorizer
from nn_index_train import build_index

DATA_CHUNKS = [
    '/datashare/2021/data_chunk_0.csv',
    '/datashare/2021/data_chunk_1.csv',
    '/datashare/2021/data_chunk_2.csv',
]

def sample_data(sample_size:int):
    sample_files = []
    i = 0
    for chunk in DATA_CHUNKS:
        status = os.system(f'head -n {sample_size} {chunk} > sample_{i}.csv')
        if status != 0:
            raise RuntimeError('failed sampling from ' + chunk)
        sample_files.append(f'sample_{i}.csv')
        i += 1

if __name__ == '__main__':
    merge(input_files=DATA_CHUNKS, output_file='_data.csv')
    preprocessing(data_path='_data.csv', output_path='data.csv')
    train_vectorizer(data_path='data.csv', output_vectors_path='vectors.pkl')
    build_index(vectors_path='vectors.pkl', output_path='nn_index.pkl')
