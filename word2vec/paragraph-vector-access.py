# Example on how to use the model
# Please compute vectors only once, cache and reuse them,
# as models may not generate reproducible results.

import gensim
from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load("../../../DATA/EML4U/amazon-reviews/amazonreviews.model")

doc = "great blue product";

tokens = gensim.utils.simple_preprocess(doc)
vector = model.infer_vector(tokens)

tokens2 = gensim.utils.simple_preprocess(doc)
vector2 = model.infer_vector(tokens)

print(tokens)
print(tokens2)
print(vector)
print(vector2)

# Subsequent calls to this function may infer different representations for the same document.
# For a more stable representation, increase the number of steps to assert a stricket convergence.
# https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec.infer_vector

# ['great', 'blue', 'product']
# ['great', 'blue', 'product']
# [-0.11509312  0.14762056 -0.03422474  0.0209586   0.04527559 -0.05443553
#  -0.01632877  0.05278325  0.02247474 -0.11322651  0.01959775  0.10782573
#   0.12373605 -0.121829    0.0274703  -0.02034174  0.09386072 -0.0250358
#   0.16765213  0.0281806  -0.02333076  0.01330676 -0.0639389  -0.17567132
#   0.04886921  0.02142672 -0.13612425  0.09088685 -0.06177525  0.01532993
#   0.05467552  0.01723739 -0.06057579 -0.07369605 -0.07298804  0.10132957
#  -0.118255   -0.16603085 -0.09985495 -0.1858428   0.05084351 -0.11533979
#   0.03381306 -0.02403254 -0.00676765 -0.09889459 -0.04066175  0.08980722
#   0.01113729  0.07410708]
# [-0.12508309  0.09653892  0.00767757 -0.00713962  0.04812891 -0.0948585
#  -0.03042169  0.0359451   0.01551943 -0.11089713  0.00889442  0.05958529
#   0.13269109 -0.08582645  0.0087731  -0.03569825  0.11176272 -0.03730286
#   0.13605452  0.01518989 -0.00719591  0.02909563 -0.02329572 -0.18949115
#   0.02656603  0.01219813 -0.12950537  0.07610258 -0.00908099  0.02442977
#   0.04653861  0.05990674 -0.09005374 -0.08324063 -0.02705938  0.08434743
#  -0.11044835 -0.14968945 -0.0840792  -0.12976211  0.01213132 -0.09632532
#   0.01482031 -0.02997352 -0.05999629 -0.12120721 -0.03231953  0.0825817
#   0.04586223  0.0754095 ]