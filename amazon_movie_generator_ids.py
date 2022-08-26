# Stores IDs (line numbers) of amazon reviews
# Based on amazon_movie_generator.py

import os
import sys
import pickle
from datetime import datetime
import time

# Config

# Original paths
movies_file = 'data/movies/movies.txt'
pickle_dir  = 'data/movies/embeddings/'

# Set your local paths, if required
#movies_file = '/home/wilke/Data/EML4U/Amazon/movies.txt'
#pickle_dir  = '/tmp/amazon-ids/'
#pickle_dir  = '/home/eml4u/EML4U/data/amazon-complete/'

id          = 0
counter     = 0
counter_max = 80 # Default is 80 based on amazon_movie_generator.py, can be used for dev
batch_size  = 100000
id_list  = []
key_list = []
ident    = []

do_extract    = True
do_check_keys = True

if do_extract:

    # Extract data

    with open(movies_file, 'r', errors='ignore') as f:
        for line in f:
            identifier = line.split(':')[0]
            if 'review/helpfulness' in identifier:
                helpfulness = line.split(':')[1].strip()
                ident.append(helpfulness)
            elif 'review/score' in identifier:
                score = line.split(':')[1].strip()
                ident.append(int(float(score)))
            elif 'review/time' in identifier:
                time_i = line.split(':')[1].strip()
                ident.append(datetime.fromtimestamp(int(time_i)))
                key_list.append(ident)
                id_list.append(id)
                ident = []

            if len(id_list) >= batch_size:
                with open(pickle_dir + '{}.pickle'.format(counter), 'wb') as handle:
                    pickle.dump((id_list, key_list), handle)

                counter += 1
                # resetting
                id_list = []
                key_list = []
                print('At step', counter, 'of', 7_911_684/batch_size)

                # Additional line for testing
                if counter_max != 80 and counter == counter_max:
                    break
            id += 1

        # save embedded data
        with open(pickle_dir + '{}.pickle'.format(counter), 'wb') as handle:
            pickle.dump((id_list, key_list), handle)
        counter += 1
        # resetting
        id_list = []
        key_list = []
        print('At step', counter, 'of', 7_911_684/batch_size)

    # Combine single files

    ids_list = []
    keys_list = []
    for counter in range(counter_max):
        with open(pickle_dir + '{}.pickle'.format(counter), 'rb') as handle:
            ids, keys = pickle.load(handle)
        ids_list.extend(ids)
        keys_list.extend(keys)

    # In-place sorting magic

    ids_list, keys_list = (list(t) for t in zip(*sorted(zip(ids_list, keys_list), key=lambda x: x[1][-1])))

    # Write

    with open(pickle_dir + 'amazon_ordered_by_time_ids{}.pickle'.format(''), 'wb') as handle:
        pickle.dump((ids_list, keys_list), handle)

    # Delete tmp files

    for i in range(counter + 1): # part files no longer needed
        os.remove(pickle_dir + '{}.pickle'.format(i))

    # Print final file info

    with open(pickle_dir + 'amazon_ordered_by_time_ids{}.pickle'.format(''), 'rb') as handle:
        ids_test, keys_test = pickle.load(handle)
        print('len(ids_test) ', len(ids_test))
        print('len(keys_test)', len(keys_test))
        print('type(ids_test) ', type(ids_test))
        print('type(keys_test)', type(keys_test))
        print('keys_test[0]', keys_test[0])
        print('ids_test[0] ', ids_test[0])



# State 2022-08-26:
# Not equal: 0 [5, datetime.datetime(1997, 8, 20, 2, 0)] ['2/2', 5, datetime.datetime(1997, 8, 20, 2, 0)]

if do_check_keys:

    with open(pickle_dir + 'amazon_ordered_by_time{}.pickle'.format(''), 'rb') as handle:
        emb_check, keys_check_emb = pickle.load(handle)
        emb_check = None
    with open(pickle_dir + 'amazon_ordered_by_time_ids{}.pickle'.format(''), 'rb') as handle:
        ids_check, keys_check_ids = pickle.load(handle)
        ids_check = None

    id = 0
    for i in range(0, len(keys_check_emb)-1):
        if keys_check_emb[i] != keys_check_ids[i]:
            print('Not equal:', i, keys_check_emb[i], keys_check_ids[i])
            break
        id += 1
    print('Checked', id, 'IDs')
    print('First element:', keys_check_emb[0])
