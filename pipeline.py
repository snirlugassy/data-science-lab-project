from utils.merge_csv import merge
from preprocessing import preprocessing
from vectorizer import train_vectorizer
from nn_index_train import build_index
from pairwise_similarities import append_related_to

DATA_CHUNKS = [
    '/datashare/2021/data_chunk_0.csv',
    '/datashare/2021/data_chunk_1.csv',
    '/datashare/2021/data_chunk_2.csv',
]

if __name__ == '__main__':
    merge(input_files=DATA_CHUNKS, output_file='_data.csv')
    preprocessing(data_path='_data.csv', output_path='data.csv')
    train_vectorizer(data_path='data.csv', output_vectors_path='vectors.pkl')
    build_index(vectors_path='vectors.pkl', output_path='nn_index.pkl')
    append_related_to(nn_index_path='nn_index.pkl', vectors_path='vectors.pkl', data_path='data.csv', output_path='data.csv')