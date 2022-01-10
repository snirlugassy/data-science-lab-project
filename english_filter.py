import re
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('normalized.csv', index_col='id', usecols=['id','industry','clean','en_prob'])

plt.hist(data['en_prob'],bins=50)
plt.ylim([0,1000])
plt.xlabel('English Probability')
plt.savefig('en_prob_hist.png')

data = data[data.en_prob > 0.75]
data.reset_index(inplace=True, drop=True)
data.rename({'clean': 'text'}, inplace=True)

industry2row = {}
for industry, group in data.groupby('industry'):
    industry2row[industry] = group.index.to_list()

data.to_csv('filtered.csv', columns=['clean', 'industry'], index=False)