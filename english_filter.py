import re
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print('Reading data')
    data = pd.read_csv(input_file, usecols=['text','industry','en_prob'])

    print('Plotting histogram')
    plt.hist(data['en_prob'],bins=50)
    plt.ylim([0,1000])
    plt.xlabel('English Probability')
    plt.savefig('en_prob_hist.png')

    print('Filtering data')
    data = data[data.en_prob > 0.75]
    data.reset_index(inplace=True, drop=True)
    data.to_csv(output_file, columns=['text', 'industry'], index=False)