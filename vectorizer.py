import csv
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_COLUMN = 'text'
MODEL_FILE = 'tfidf.sklearn.pkl'
VECTORS_FILE = 'vectors.pkl'


def read_corpus(file_path):
    with open(file_path, 'r') as data_file:
        reader = csv.reader(data_file)
        header = next(reader)
        text_index = header.index(TEXT_COLUMN)
        for row in reader:
            yield row[text_index]

def train_vectorizer(data_path, output_vectors_path=VECTORS_FILE, output_model_path=MODEL_FILE, max_df=0.7):
    csv.field_size_limit(sys.maxsize)
    vectorizer = TfidfVectorizer(max_df=max_df)
    
    print('Fitting vectorizer')
    X = vectorizer.fit_transform(read_corpus(data_path))
    print("tf-idf matrix shape=", X.shape)

    print(f'Saving tf-idf vectorizer to {output_model_path}')
    with open(output_model_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f'Saving tf-idf matrix to {output_vectors_path}')
    with open(output_vectors_path, "wb") as f:
        pickle.dump(X, f)

if __name__ == '__main__':
    train_vectorizer(sys.argv[1], VECTORS_FILE, MODEL_FILE, 0.7)