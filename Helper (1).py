import csv
import pickle
import re
import sys
import time

import gensim
import pandas as pd
import scipy.sparse
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

STOPWORDS = stopwords.words('english')
HTML_TAG = re.compile('<.*?>')
from sklearn.cluster import MiniBatchKMeans



class DataProcessing:
    def __init__(self, input_files: list, output_file: str, nrows=20000):
        self.files = input_files

    def run(self):
        pass


class TextNormalizer:
    def __init__(self):
        self.__stemmer = SnowballStemmer('english', ignore_stopwords=False)

    def _stem(self, t):
        return self.__stemmer.stem(t)

    def normalize(self, text):
        if isinstance(text, str):
            text = re.sub(HTML_TAG, ' ', text)
            text = re.sub('\W', ' ', text)
            text = re.sub('\d+', ' ', text)
            text = re.sub('\s+', ' ', text)
            text = text.lower().strip()
            text = str.join(' ', [self._stem(x) for x in text.split(' ') if len(x) > 1 and x not in STOPWORDS])
            return text
        return ''

def build_clean_text_files():
    for i in range(3):
        data_path = "/datashare/2021/data_chunk_" + str(i) + ".csv"
        output_path = "chunk_" + str(i) + "_stem.csv"

        print('Reading data from', data_path)
        data = pd.read_csv(data_path, index_col='id')
        normalizer = TextNormalizer()

        print('Cleaning text')
        data['clean'] = data.text.apply(lambda t: normalizer.normalize(t))

        print('Calculating English probability')
        data['en_prob'] = data.clean.apply(lambda t: len(re.findall('[A-Za-z\s]', t)) / (len(t) + 1))

        data.drop(columns=['text', 'url', 'website', 'linkedin'], inplace=True)

        print('Saving to', output_path)
        data.to_csv(output_path)


def build_dictionary():
    data = pd.read_csv("chunk_0_stem.csv")
    dct = corpora.Dictionary()
    counter = 0
    data = data.head(5000)
    for line in data['clean']:
        if counter % 1000 == 0:
            print(counter)
        text = [line.split()]
        dct.add_documents(text, prune_at=2000000000)
        counter += 1

    dct.save("dictionary_check")
    print("dictionary saved")



def compute_tfidf_example():
    import gensim.downloader as api
    from gensim.models import TfidfModel
    from gensim.corpora import Dictionary
    dct = gensim.corpora.dictionary.Dictionary.load("dictionary_check")
    vocabulary = [val for val in dct.values()]
    corpus = []
    data = pd.read_csv("chunk_0_stem.csv")
    data = data.head(5000)
    for line in data['clean']:
        corpus.append(line)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(X, f)

def kmeans_clustering():
    with open("vectorizer.pkl", "rb") as f:
        X = pickle.load(f)

    kmeans = MiniBatchKMeans(n_clusters=10, random_state=0, batch_size=1000, max_iter=10).fit(X)
    centers = kmeans.cluster_centers_
    labels = dict()
    for i in range(10):
        labels[i] = []
    for idx, label in enumerate(kmeans.labels_):
        labels[label].append(idx)

    with open("kmeans_and_labels.pkl", "wb") as f:
        pickle.dump((kmeans, labels), f)


if __name__ == '__main__':
    #build_clean_text_files()
    #build_dictionary()
    #compute_tfidf_example()
    #kmeans_clustering()

    with open("kmeans_and_labels.pkl", "rb") as f:
        kmeans, labels = pickle.load(f)

    i = 0

    from scipy.spatial.distance import cdist, pdist
    from sklearn.metrics.pairwise import pairwise_distances

    with open("vectorizer.pkl", "rb") as f:
        X = pickle.load(f)

    row_to_industry_dict_from_snir = dict()
    for i in range(5000):
        if i < 200:
            row_to_industry_dict_from_snir[i] = "a"
        else:
            row_to_industry_dict_from_snir[i] = "b"

    #row_to_label_dict = dict()
    labels_copy = labels.copy()
    related_of = dict()
    for key, vals in labels.items():
        current_other_indices = []
        current_industry_indices = []
        used_industries = []
        for val in vals:
            if row_to_industry_dict_from_snir[val] not in used_industries:
                used_industries.append(row_to_industry_dict_from_snir[val])
                for val_copy in labels_copy[key]:
                    if row_to_industry_dict_from_snir[val] != row_to_industry_dict_from_snir[val_copy]:
                        current_other_indices.append(val_copy)
                    else:
                        current_industry_indices.append(val_copy)
                tfidf_industry = X[current_industry_indices, :] #csr
                tfidf_other = X[current_other_indices, :] #csr
                cosine_values = pairwise_distances(X=tfidf_industry, Y=tfidf_other, metric='cosine')
                cosine_values = scipy.sparse.csr_matrix(cosine_values)
                indices = cosine_values.argmax(axis=1)
                for i, index in enumerate(current_industry_indices):
                    related_of[index] = current_other_indices[indices[i,0]]
                print(cosine_values)





