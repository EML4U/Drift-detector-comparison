import pickle
import sys
import os

if len(sys.argv) < 3:
    print('Need filename as parameter!\nExample useage: python3 delete_specific_result.py some_result_file.pickle detector_to_delete')
    exit(1)
    
filename = sys.argv[1]
detector_to_delete = sys.argv[2]

with open(filename, 'rb') as handle:
    results = pickle.load(handle)
    
    
if 'twitter_diff_dist' in filename:
    for res in results:
        for mode in res:
            res[mode].pop(detector_to_delete)
else:
    for mode in results:
        results[mode].pop(detector_to_delete)


with open(filename, 'wb') as handle:
    pickle.dump(results, handle)