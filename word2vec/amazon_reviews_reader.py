# Downloader and iterator to read Amazon movie reviews
# https://snap.stanford.edu/data/movies.txt.gz
# https://snap.stanford.edu/data/web-Movies.html
#
# Download example:
# from amazon_reviews_reader import AmazonReviewsReader
# AmazonReviewsReader.download("/tmp")
#
# Iterator
# Modes:
# fields, text (summary and text), tokens (words), tagdoc
# Example:
# from amazon_reviews_reader import AmazonReviewsReader
# for item in AmazonReviewsReader("/tmp/movies.txt.gz", "fields", max_docs=3):
#     print(item)

from datetime import datetime
import gensim
import gzip
import os
import subprocess

# Stream corpus (memory efficient)
# See: https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-one-document-at-a-time
class AmazonReviewsReader:

    # Downloads file if not available
    #
    # https://snap.stanford.edu/data/web-Movies.html
    # 3321791660 bytes / 3 GB
    @staticmethod
    def download(directory, url="https://snap.stanford.edu/data/movies.txt.gz"):
        file_path = os.path.join(directory, url.rsplit('/', 1)[1])
        if not os.path.isfile(file_path):
            print("Download", url, directory)
        # https://www.gnu.org/software/wget/manual/wget.html#Download-Options
        # -c  --continue
        # -nv --no-verbose
        # -P  --directory-prefix=prefix
        subprocess.run(["wget", "-c", "-nv", "-P", directory, url])


    def __init__(self, file, mode, max_docs=-1, min_year=-1, max_year=-1, min_score=-1, max_score=-1):
        self.file = file
        self.mode = mode
        self.max_docs = max_docs
        self.min_year = min_year
        self.max_year = max_year
        self.min_score = min_score
        self.max_score = max_score

    def __iter__(self):
        c = 0
        i = 0
        with gzip.open(self.file, 'rb') as f:
            for line in f:
                if not line.strip():
                    continue
                line_spilt = line.decode(encoding='iso-8859-1').split(':', 1)

                # First key of every entry -> reset values
                if line_spilt[0] == "product/productId":
                    self.exclude = False
                    self.entry = {}
                    
                # Get key/value
                self.entry[line_spilt[0].split('/', 1)[1]] = line_spilt[1].strip()

                # Filter by score
                if self.min_score!=-1 and self.max_score!=-1 and line_spilt[0] == "review/score":
                    score = float(line_spilt[1].strip())
                    if(self.min_score != -1 and score < self.min_score):
                        self.exclude = True
                    elif(self.max_score != -1 and score > self.max_score):
                        self.exclude = True

                # Filter by year
                elif self.min_year!=-1 and self.max_year!=-1 and line_spilt[0] == "review/time":
                    year = datetime.fromtimestamp(int(line_spilt[1])).year
                    if(self.min_year != -1 and year < self.min_year):
                        self.exclude = True
                    elif(self.max_year != -1 and year > self.max_year):
                        self.exclude = True

                # Last key -> Process data
                elif line_spilt[0] == "review/text":
                    
                    # Filter
                    i += 1
                    if(self.exclude):
                        continue
                    
                    # Max number of docs
                    c += 1
                    if(self.max_docs != -1 and c > self.max_docs):
                        break
                    
                    # Mode fields
                    if(self.mode == "fields"):
                        self.entry["number"] = i
                        yield self.entry
                        continue
                    
                    # Mode text
                    if(self.mode == "text"):
                        yield self.entry["summary"] + " " + self.entry["text"]
                        continue
                    
                    # Mode tokens
                    tokens = gensim.utils.simple_preprocess(self.entry["summary"] + " " + self.entry["text"])
                    if(self.mode == "tokens"):
                        yield tokens
                        continue
                    
                    # Mode tagdoc
                    if(self.mode != "tagdoc"):
                        raise ValueError("Unknown mode")
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [c])