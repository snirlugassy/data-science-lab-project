from mrjob.job import MRJob

HEADER = 'text,industry,related_to,related_industry'
TEXT = 0
INDUSTRY = 1
RELATED_TO = 2
RELATED_INDUSTRY = 3

class IndustryMapReduce(MRJob):
    def mapper(self, _, row):
        if not row.startswith(HEADER):
            company = row.split(',')
            # skip invalid line seperation
            if not (company[INDUSTRY].isalnum() or company[RELATED_INDUSTRY].isalnum()):
                yield ((company[INDUSTRY], company[RELATED_INDUSTRY]), 1)

    def reducer(self, key, values):
        yield key, sum(values)

if __name__ == '__main__':
    IndustryMapReduce.run()