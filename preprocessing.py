import re
import sys
import pandas as pd
import numpy as np


class DataProcessing:
    def __init__(self, input_files: list[str], output_file: str, nrows=20000):
        self.files = input_files
    
    def run(self):
        pass


class TextNormalization:
    @staticmethod
    def normalize(text):
        return text


if __name__ == '__main__':
    data_path = sys.argv[1]
    data = pd.read_csv(data_path, nrows=10000)
