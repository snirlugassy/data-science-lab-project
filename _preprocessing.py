import re
import sys
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

STOPWORDS = {x:1 for x in stopwords.words('english')}
HTML_TAG = re.compile('<.*?>')


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
            text = str.join(' ', [x for x in text.split(' ') if len(x) > 2 and len(x) < 15 and not STOPWORDS.get(x)])   
            return text
        return ''

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_path = sys.argv[2]

    print('Reading data from', data_path)
    data = pd.read_csv(data_path, index_col='id').query('country=="united states"')
    normalizer = TextNormalizer()

    print('Cleaning text')
    data['clean'] = data.text.apply(lambda t: normalizer.normalize(t))

    print('Calculating English probability')
    data['en_prob'] = data.clean.apply(lambda t: len(re.findall('[A-Za-z\s]', t)) / (len(t) + 1))

    data.drop(columns=['text', 'url', 'website', 'linkedin'], inplace=True)

    print('Saving to', output_path)
    data.to_csv(output_path)