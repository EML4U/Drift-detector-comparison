import sys
import os

# This is designed to run all the data generation scripts in the required order

modes = ['bert_768', 'bow_50', 'bow_768']

# starting with amazon
print('Generating amazon data')
for mode in modes:
    print('Now generating {}'.format(mode))
    os.system('python3 generator_amazon_movie_drift_data.py {mode}'.format(mode=mode))
    os.system('python3 generator_amazon_movie_same_dist.py {mode}'.format(mode=mode))
    os.system('python3 generator_amazon_movie_different_classes.py {mode}'.format(mode=mode))
    
    

# then twitter
print('Generating twitter data')
for mode in modes:
    print('Now generating {}'.format(mode))
    os.system('python3 generator_twitter_diff_classes.py {mode}'.format(mode=mode))
    os.system('python3 generator_twitter_diff_dists.py {mode}'.format(mode=mode))
    os.system('python3 generator_twitter_same_dist.py {mode}'.format(mode=mode))
    os.system('python3 generator_twitter_drift_data.py {mode}'.format(mode=mode))









