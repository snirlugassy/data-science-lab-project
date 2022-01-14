import csv
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_COLUMN = 'text'

def read_corpus(file_path):
    with open(file_path, 'r') as data_file:
        reader = csv.reader(data_file)
        header = next(reader)
        text_index = header.index(TEXT_COLUMN)
        for row in reader:
            yield row[text_index]

if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)
    input_file = sys.argv[1]
    vectorizer = TfidfVectorizer(max_df=0.7)
    
    print('Fitting vectorizer')
    X = vectorizer.fit_transform(read_corpus(input_file))
    print("tf-idf matrix shape=", X.shape)

    print('Saving tf-idf vectorizer to tfidf.sklearn.pkl')
    with open('tfidf.sklearn.pkl', "wb") as f:
        pickle.dump(vectorizer, f)

    print('Saving tf-idf matrix to vectors.pkl')
    with open('vectors.pkl', "wb") as f:
        pickle.dump(X, f)

# if __name__ == '__main__':
#     csv.field_size_limit(sys.maxsize)
#     input_file = sys.argv[1]
#     print('loading vectorizer')
#     vectorizer = pickle.load(open('tfidf.sklearn.pkl','rb'))

#     print('transforming text')
#     X = vectorizer.transform(read_corpus(input_file))

#     print('Saving tf-idf matrix to vectors.pkl')
#     with open('vectors.pkl', "wb") as f:
#         pickle.dump(X, f)
