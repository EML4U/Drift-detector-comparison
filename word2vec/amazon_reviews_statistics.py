# Statistics for Amazon movie reviews

import yaml
import os.path
from amazon_reviews_reader import AmazonReviewsReader
from datetime import datetime

config         = yaml.safe_load(open("../config.yaml", 'r'))
amazon_gz_file = os.path.join(config["AMAZON_MOVIE_REVIEWS_DIRECTORY"], "movies.txt.gz")
max_docs       = -1

years_scores = dict()
docs = 0
text_length    = 0
summary_length = 0
text_words    = 0
summary_words = 0

for item in AmazonReviewsReader(amazon_gz_file, "fields", max_docs=max_docs):
    year = datetime.fromtimestamp(int(item["time"])).year
    score = int(float(item['score']))
    key = str(year) + "_" + str(score)
    years_scores[key] = years_scores.get(key , 0) + 1
     
    docs = docs + 1
    
    text_length = text_length + len(item['text'])
    summary_length = summary_length + len(item['summary'])
    
    text_words = text_words + len(item['text'].split())
    summary_words = summary_words + len(item['summary'].split())

for k in sorted(years_scores.keys()):
    print(k, years_scores[k])

print("processed items    ", docs)
print("mean text length   ", text_length/docs)
print("mean summary length", summary_length/docs)
print("mean text words    ", text_words/docs)
print("mean summary words ", summary_words/docs)

# 1997_1 6
# 1997_2 1
# 1997_3 8
# 1997_4 29
# 1997_5 64
# 1998_1 191
# 1998_2 262
# 1998_3 442
# 1998_4 797
# 1998_5 3313
# 1999_1 4844
# 1999_2 3631
# 1999_3 6458
# 1999_4 14178
# 1999_5 49866
# 2000_1 19944
# 2000_2 17808
# 2000_3 30907
# 2000_4 73314
# 2000_5 192002
# 2001_1 24221
# 2001_2 20320
# 2001_3 35395
# 2001_4 79152
# 2001_5 189638
# 2002_1 25311
# 2002_2 22641
# 2002_3 37798
# 2002_4 84276
# 2002_5 198712
# 2003_1 25734
# 2003_2 24183
# 2003_3 43323
# 2003_4 90527
# 2003_5 205916
# 2004_1 41016
# 2004_2 33117
# 2004_3 60489
# 2004_4 119160
# 2004_5 257603
# 2005_1 54744
# 2005_2 40868
# 2005_3 71012
# 2005_4 138000
# 2005_5 308080
# 2006_1 49049
# 2006_2 37992
# 2006_3 66128
# 2006_4 135581
# 2006_5 311252
# 2007_1 49521
# 2007_2 40205
# 2007_3 75239
# 2007_4 167632
# 2007_5 452009
# 2008_1 56076
# 2008_2 40138
# 2008_3 74057
# 2008_4 161693
# 2008_5 412870
# 2009_1 59099
# 2009_2 39680
# 2009_3 73178
# 2009_4 149771
# 2009_5 422403
# 2010_1 65343
# 2010_2 41430
# 2010_3 70279
# 2010_4 142000
# 2010_5 426248
# 2011_1 72957
# 2011_2 45767
# 2011_3 72055
# 2011_4 148457
# 2011_5 465918
# 2012_1 81276
# 2012_2 47356
# 2012_3 74826
# 2012_4 150248
# 2012_5 484650
# processed items     7911684
# mean text length    956.7743798918157
# mean summary length 27.669462658013135
# mean text words     167.43782840669573
# mean summary words  4.815367625906191