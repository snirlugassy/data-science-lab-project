import sys
import csv
import re

from nltk.corpus import stopwords
STOPWORDS = {x:1 for x in stopwords.words('english')}

HTML_TAG = re.compile('<.*?>')
TEXT = 'text'
INDUSTRY = 'industry'
COUNTRY = 'country'
LATIN = 'latin'
USA = 'united states'
LATIN_THRESHOLD = 0.9

def normalize_text(text):
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
    csv.field_size_limit(sys.maxsize)

    data_path = sys.argv[1]
    output_path = sys.argv[2]

    print('Reading data from', data_path)
    reader = csv.DictReader(open(data_path, 'r'))
    writer = csv.DictWriter(open(output_path, 'w'), fieldnames=[TEXT, INDUSTRY])
    writer.writeheader()
    
    line_count = 0
    for line in reader:
        line_count += 1
        print(f'Line {line_count}', end='\r')
        if line[COUNTRY] == USA:
            _text = normalize_text(line[TEXT])
            _latin = len(re.findall('[A-Za-z\s]', _text)) / (len(_text) + 1)
            if _latin > LATIN_THRESHOLD:
                writer.writerow({
                    TEXT: _text,
                    INDUSTRY: line[INDUSTRY]
                })
