import re
import sys
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    csv.field_size_limit(sys.maxsize)

    print('Reading data')
    data_path = sys.argv[1]
    latin_usa_frac = []
    latin_frac = []
    line_count = 0
    for line in csv.DictReader(open(data_path, 'r')):
        line_count += 1
        print(f'Line {line_count}', end='\r')
        text = line['text']
        _latin = len(re.findall('[A-Za-z\s]', text)) / (len(text) + 1)
        latin_frac.append(_latin)
        if line['country'] == 'united states':
            latin_usa_frac.append(_latin)

    print('Saving histograms')
    plt.figure()
    plt.title('Percentage of Latin letters in text histogram')
    plt.hist(latin_frac, bins=50)
    plt.ylim([0,1000])
    plt.xlabel(f'% Latin Letters (All countries, sample size={len(latin_frac)})')
    plt.savefig('latin_dist_all_countries.png')

    plt.figure()
    plt.title('Percentage of Latin letters in text histogram')
    plt.hist(latin_usa_frac, bins=50)
    plt.ylim([0,1000])
    plt.xlabel(f'% Latin Letters (Only USA, sample size={len(latin_usa_frac)})')
    plt.savefig('latin_dist_usa.png')

    # print('Filtering data')
    # data = data[data.en_prob > 0.75]
    # data.reset_index(inplace=True, drop=True)
    # data.to_csv(output_file, columns=['text', 'industry'], index=False)