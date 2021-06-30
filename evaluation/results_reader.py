# Usage example:
# Import results reader functions
#from results_reader import (read_results, print_results, get_result_ids)
# Read results:
#results = read_results("../data/results/", print_info=True)
# Get sorted IDs of results:
#ids, modes, detectors, keys = get_result_ids(results)
#print(ids, modes, detectors, keys, sep="\n")
# Print overview of runtimes for LSDD and BoW-50:
#print_results(results, ids=[], modes=["bow_50"], detectors=["lsdd"], keys=["time_detect", "time_fit"])


import collections
from os import listdir
from os.path import isfile, join
import pickle


# Reads results files
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
                id_.ljust(25),
                results[id_]["result_pickle"].ljust(50),
                str(type(results[id_]["data"])).ljust(10),
                len(results[id_]["data"])
            )

    return results


# Prints overview of loaded results
def print_results(results, ids=[], modes=[], detectors=[], keys=[]):
    for id_ in ids if ids else results:
        data = results[id_]["data"]
        for mode in modes if modes else data:
            for detector in detectors if detectors else data[mode]:
                for key in keys if keys else data[mode][detector]:
                    print(id_.ljust(23), mode.ljust(9), detector.ljust(5), key.ljust(13), end="")
                    if key in data[mode][detector] and data[mode][detector][key]:
                        print(str(type(data[mode][detector][key])).ljust(17), end="")
                        if isinstance(data[mode][detector][key], list) or isinstance(data[mode][detector][key], dict):
                            print(str(len(data[mode][detector][key])), "(len)")
                        else:
                            print(str(data[mode][detector][key])[:10])
                    else:
                        print("None")


# Gets lists of sorted IDs
def get_result_ids(results, ids=[], modes=[], detectors=[], keys=[]):
    ids_ = set()
    modes_ = set()
    detectors_ = set()
    keys_ = set()
    for id_ in ids if ids else results:
        data = results[id_]["data"]
        for mode in modes if modes else data:
            for detector in detectors if detectors else data[mode]:
                for key in keys if keys else data[mode][detector]:
                    if key in data[mode][detector] and data[mode][detector][key]:
                        ids_.add(id_)
                        modes_.add(mode)
                        detectors_.add(detector)
                        keys_.add(key)
    return sorted(ids_), sorted(modes_), sorted(detectors_), sorted(keys_)
