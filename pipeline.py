import os
import sys
import subprocess
from typing import Iterable

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

def merge_data_chunks(chunks:Iterable, output_file):
    _merge_script_path = os.path.join('utils', 'merge_csv.py')
    _chunks = str.join(' ', chunks)
    status = os.system(f'python {_merge_script_path} {_chunks} {output_file}')
    if status != 0:
        raise RuntimeError(f'Failed to merge {_chunks}')

def run_text_processing(input_file:str, output_file:str):
    assert input_file != output_file
    status = os.system(f'python preprocessing.py {input_file} {output_file}')
    if status != 0:
        raise RuntimeError(f'Failed to preprocess {input_file}')

def run_vectorizer(input_file:str):
    status = os.system(f'python vectorizer.py {input_file}')
    if status != 0:
        raise RuntimeError(f'Failed to run preprocessing on {input_file}')

def run_clustering(data_file:str):
    status = os.system(f'python clustering.py {data_file}')
    if status != 0:
        raise RuntimeError(f'Failed to run clustering on {data_file}')

def run_pairwise_similarities(input_file:str):
    pass

def run_map_reduce(input_file:str):
    pass

if __name__ == '__main__':
    # samples = sample_data(100000)
    # merge_data_chunks(DATA_CHUNKS, '_data.csv')
    run_text_processing('_data.csv', 'data.csv')
    run_vectorizer('data.csv')