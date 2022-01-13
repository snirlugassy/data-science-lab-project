import sys
import csv
import re
from preprocessing import TextNormalizer

TEXT = 'text'
INDUSTRY = 'industry'
EN_PROB = 'en_prob'

if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)

    data_path = sys.argv[1]
    output_path = sys.argv[2]

    print('Reading data from', data_path)
    reader = csv.DictReader(open(data_path, 'r'))
    writer = csv.DictWriter(open(output_path, 'w'), fieldnames=[TEXT, INDUSTRY, EN_PROB])
    writer.writeheader()
    
    normalizer = TextNormalizer()
    line_count = 0
    for line in reader:
        line_count += 1
        print(f'Line {line_count}', end='\r')
        _text = normalizer.normalize(line[TEXT])
        writer.writerow({
            TEXT: _text,
            INDUSTRY: line[INDUSTRY],
            EN_PROB: len(re.findall('[A-Za-z\s]', _text)) / (len(_text) + 1)
        })
