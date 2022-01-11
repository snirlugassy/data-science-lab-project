# Data science lab project

1. preprocessing.py
Preprocess the company text, using the following transformations:
- Remove HTML tags
- Remove any special symbols (non characters or non digits)
- Remove any number
- Convert to lower case
- Remove stopwords
- ??? Stemming using [SnowballStemmer](https://snowballstem.org/)

The transformed text is added to a new column called `clean`.

Afterwards, The column `en_prob` is added, definition:
$$ P(text \in English) \approx en\_prob :=  {{|\{c \in [a,z]\cup[A,Z] : c \in text \}|} \over {|text|+1}}$$

Expected columns in input CSV:
- id
- text
- industry

Usage template:
```
python3 preprocessing.py /path/to/input.csv /path/to/output.csv
```


2. vectorizer.py
Expected columns in input CSV:
- id
- clean (should be renamed to text or tokens)
- industry

```
python3 preprocessing.py /path/to/preprocessed.csv /path/to/output.???
```


3. similarity.py
```
python3 preprocessing.py /path/to/vectors /path/to/output
```

4. map_reduce.py
5. statistics.py