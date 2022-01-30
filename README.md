# Data science lab project

Our goal is to process a large corpus scraped from websites of companies and to construct a weighted directed graph that represents the relations between different industries, we will refer to this graph as Industry Network.

The Industry network brings insights about the corpus and underlying semantic relations between industries, moreover, it can be later used in other tasks such as clustering or recommendation systems.

Our pipeline can be easily transferred to other domains such as products, twits, or academic papers, to map the categories of the domain.

### Setup

Creating a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Installing pysparnn:
```
git clone https://github.com/facebookresearch/pysparnn.git
cd pysparnn 
pip install -r requirements.txt 
python setup.py install
cd ..
```

Installing requirements:
```
pip install -r requirements.txt
```

### Using the pipeline

We created `pipeline.py` to automated the whole pipeline process instead of manually running different scripts.
Modify the path for the data chunks under `pipeline.py`, the current paths are:
```
DATA_CHUNKS = [
    '/datashare/2021/data_chunk_0.csv',
    '/datashare/2021/data_chunk_1.csv',
    '/datashare/2021/data_chunk_2.csv',
]
```

Run:
```
python pipeline.py
```

### Network analysis
The network analysis is under `industry_graph.ipynb`, assuming you have `data.csv` populated with 'related_to' and 'related_industry' columns.
Open the notebook and run all cells.
