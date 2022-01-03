import re
import sys
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')
HTML_TAG = re.compile('<.*?>')

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
        text = re.sub(HTML_TAG, ' ', text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\d+', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.lower().strip()
        text = str.join(' ', [self._stem(x) for x in text.split(' ') if len(x) > 1 and x not in STOPWORDS])   
        return text

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    data = pd.read_csv(data_path, index_col='id')
    normalizer = TextNormalizer()
    data['clean'] = data.text.apply(lambda t: normalizer.normalize(t))
    data['en_prob'] = data.clean.apply(lambda t: len(re.findall('[A-Za-z\s]', t)) / len(t))
    data.drop(columns=['text', 'url', 'website', 'linkedin'], inplace=True)
    data.to_csv(output_path)