from typing import Iterable

dataset = api.load("text8")
dct = Dictionary(dataset)  # fit dictionary
corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format

model = TfidfModel(corpus)  # fit model
vector = model[corpus[0]] 

class Vectorizer:
    def fit(self, corpus: Iterable[str]):
        pass

    def transform(self, corpus: Iterable[str]):
        pass
