# EML4U Drift Detector Comparison


## Dataset

### Amazon movie reviews

Download the ordered by time embeddings from [Semalytix Google Drive](https://drive.google.com/drive/folders/1CRwXsKj8984PF0Cg7wpVu7Ib8S0SATof).  
Otherwise generate it yourself with the amazon_movie_generator.py (Might take around 2 days on a Quadro GPU).

![](figures/amazon-overview/amazon-overview.svg)


## Files

### Data preparation

- amazon_movie_generator.py
    - Reads source lines and generates pickle file.
    - In: https://snap.stanford.edu/data/movies.txt.gz
    - Out: data/movies/embeddings/{}.pickle
    - In 2: data/movies/embeddings/{}.pickle
    - Out 2: data/movies/embeddings/**amazon_ordered_by_time{}.pickle**
- amazon_movie_sorter.py
    - Sorts datasets by helpfulness.score.time and saves it along with text.
    - In: data/movies/movies.txt
    - Out: data/movies/embeddings/**amazon_raw.pickle**
- create_small_embedded.py
    - In: data/movies/embeddings/amazon_raw.pickle
    - Out: data/movies/embeddings/**amazon_small.pickle**

### Drift injection

- amazon_movie_generator_drift_all.py
    - Gathers data of 4. year.
      Adds negative words with percentages 0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1 and computes embeddings.
    - In: data/movies/embeddings/amazon_raw.pickle
    - In: data/sentiment_words/negative-words.txt
    - Out: data/movies/embeddings/**amazon_small_gradual_drift.pickle**
- amazon_movie_generator_drift.py
    - Similar to file above.
    - In: data/movies/embeddings/amazon_raw.pickle
    - In: data/sentiment_words/negative-words.txt
    - Out: data/movies/embeddings/**amazon_small_drift.pickle**

### Ground truth

- 'Ground Truth calc.ipynb'
    - Uses Classifier
    - In: data/movies/embeddings/**amazon_small.pickle**
- Classifier.py
    - Used by 'Ground Truth calc.ipynb'

### Drift experiments

- 'Drift experiments Alibi Kolmogorov-Smirnov.ipynb'
- 'Drift experiments Cosine.ipynb'
- 'Drift experiments FCIT.ipynb'
- 'Drift experiments KernelTwosample.ipynb'
