# Iterator for reading summary and text fields from
# https://snap.stanford.edu/data/movies.txt.gz
# Returns values for modes: text, tokens, tagdoc

import gzip
from datetime import datetime
import gensim

# Stream corpus (memory efficient)
# See: https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-one-document-at-a-time
class AmazonReviewsReader:
    
    def __init__(self, file, mode, max_docs=-1, min_year=-1, max_year=-1, min_score=-1, max_score=-1):
        self.file = file
        self.mode = mode
        self.max_docs = max_docs
        self.min_year = min_year
        self.max_year = max_year
        self.min_score = min_score
        self.max_score = max_score

    def __iter__(self):
        i = 0
        with gzip.open(self.file, 'rb') as f:
            for line in f:
                line_spilt = line.decode(encoding='iso-8859-1').split(':')
                if "review/userId" in line_spilt[0]:
                    self.exclude = False
                if "review/score" in line_spilt[0]:
                    score = float(line_spilt[1].strip())
                    if(self.min_score != -1 and score < self.min_score):
                        self.exclude = True
                    elif(self.max_score != -1 and score > self.max_score):
                        self.exclude = True
                if "review/time" in line_spilt[0]:
                    year = datetime.fromtimestamp(int(line_spilt[1])).year
                    if(self.min_year != -1 and year < self.min_year):
                        self.exclude = True
                    elif(self.max_year != -1 and year > self.max_year):
                        self.exclude = True
                if "review/summary" in line_spilt[0]:
                    text = line_spilt[1]
                if "review/text" in line_spilt[0]:
                    if(self.exclude):
                        continue
                    
                    i += 1
                    if(self.max_docs != -1 and i > self.max_docs):
                        break
                           
                    text += " " + line_spilt[1]
                    if(self.mode == "text"):
                        yield text
                        text = ""
                        continue
                    
                    tokens = gensim.utils.simple_preprocess(text)
                    text = ""
                    if(self.mode == "tokens"):
                        yield tokens
                        continue
                    
                    if(self.mode != "tagdoc"):
                        raise ValueError("Unknown mode")
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])