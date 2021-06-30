# Usage example:
# Import results reader functions
#from results_reader import (read_results, print_overview)
# Set directory with results files:
#results_directory = "../data/results/"
# Read results:
#results = read_results(results_directory, print_info=True)
# Print overview of runtimes:
#print_overview(results, ids=[], modes=[], detectors=[], keys=["time_detect", "time_fit"])


import collections
from os import listdir
from os.path import isfile, join
import pickle


def read_results(results_directory, print_info=True):
    results = collections.OrderedDict()

    # get files and ids
    for name in sorted(listdir(results_directory)):
        path = join(results_directory, name)
        if isfile(path):
            results[name[:-7]] = {"result_pickle":path}

    # add data
    results_twitter_diff_dist = collections.OrderedDict() # additional results
    for id_ in results:
        with open(results[id_]["result_pickle"], 'rb') as handle:
            if id_ == "twitter_diff_dist": # 3 results as tuple
                data = pickle.load(handle)
                results_twitter_diff_dist[id_ + "_124"] = { "result_pickle":path , "data":data[0] }
                results_twitter_diff_dist[id_ + "_192"] = { "result_pickle":path , "data":data[1] }
                results_twitter_diff_dist[id_ + "_480"] = { "result_pickle":path , "data":data[2] }
            else:
                results[id_]["data"] = pickle.load(handle)
    results.update(results_twitter_diff_dist) # join results
    del results["twitter_diff_dist"] # remove obsolete data

    # print info
    if(print_info):
        for id_ in results: 
            print(
                id_, " ",
                results[id_]["result_pickle"], " ",
                type(results[id_]["data"]), " ",
                len(results[id_]["data"])
            )

    return results


def print_overview(results, ids=[], modes=[], detectors=[], keys=[]):
    for id_ in ids if ids else results:
        data = results[id_]["data"]
        for mode in modes if modes else data:
            for detector in detectors if detectors else data[mode]:
                for key in keys if keys else data[mode][detector]:
                    print(id_.ljust(23), mode.ljust(9), detector.ljust(5), key.ljust(12), end="")
                    if key in data[mode][detector] and isinstance(data[mode][detector][key], list):
                        print("", str(len(data[mode][detector][key])).ljust(4), end="")
                    else:
                        print("     ", end="")
                    print(results[id_]['result_pickle'], end="")
                    print()
