class DataProcessing:
    def __init__(self, input_files:list[str], output_file:str, nrows=20000):
        self.files = input_files
    
    def run(self):
        pass


class TextNormalization:
    @staticmethod
    def normalize(text):
        return text
