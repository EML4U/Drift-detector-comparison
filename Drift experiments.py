# Raw version of notebook

import yaml
import os.path
import pickle

# Set data paths
config          = yaml.safe_load(open("config.yaml", "r"))
bow_50_file  = os.path.join(config["EMBEDDINGS_DIRECTORY"], "amazon_drift_bow_50.pickle")
bow_768_file = os.path.join(config["EMBEDDINGS_DIRECTORY"], "amazon_drift_bow_768.pickle")
results_file = os.path.join(config["EXPERIMENTS_DIRECTORY"], "results_a")
print("bow_50_file", bow_50_file)
print("bow_768_file", bow_768_file)

# Load data
data = {}
with open(bow_50_file, "rb") as handle:
    data["bow_50"] = pickle.load(handle)
print("Samples:", len(data["bow_50"]['orig'][0]), len(data["bow_50"]['drifted'][0][0]), len(data["bow_50"]['train'][0]))
with open(bow_768_file, "rb") as handle:
    data["bow_768"] = pickle.load(handle)
print("Samples:", len(data["bow_768"]['orig'][0]), len(data["bow_768"]['drifted'][0][0]), len(data["bow_768"]['train'][0]))



if(False):
    print_model = data["bow_50"]
    print(type(print_model), len(print_model))
    for key, value in print_model.items() :
        print (key, type(value), len(value))
        for i in range(len(value)) :
            print (value[i][0])
            print()

            
            
results = {}



# Load previous results
if os.path.isfile(results_file):
    with open(results_file, "rb") as handle:
        results = pickle.load(handle)


            
import time

# Call fit funtion, if not already in results
def default_fit(detector_id, detector, data_id, data, results, force_run):
    if(data_id in results and detector_id in results[data_id] and not force_run):
        return
    
    # Reset results
    results_detector = {}
    
    time_begin = time.time()
        
    detector.fit(data)
    
    results_detector["time_fit"] = time.time() - time_begin
    
    if(data_id not in results):
        results[data_id] = {}
    results[data_id][detector_id] = results_detector

# Compute predictions, if not already in results
def default_detect(detector_id, detector, data_id, data, results, force_run):
    if(data_id in results and detector_id in results[data_id] and
       "predictions" in results[data_id][detector_id] and not force_run):
        return
    
    # Get previous results
    if(data_id in results and detector_id in results[data_id]):
        results_detector = results[data_id][detector_id]
    else:
        results_detector = {}
    
    time_begin = time.time()
    
    results_detector["predictions"] = []
    print(data_id, detector_id, end=" ")
    for p in data:
        results_detector["predictions"].append(detector.predict_proba(p))
        print(len(p) , end=" ")
    print()

    results_detector["time_detect"] = time.time() - time_begin

    if(data_id not in results):
        results[data_id] = {}
    results[data_id][detector_id] = results_detector
    
    
    
# Save results
def save_results(results_file, results):
    with open(results_file, "wb") as handle:
        pickle.dump(results, handle)

        

if(False):
    from detectors.AlibiMMDDetector import AlibiMMDDetector
    detector_id = "AlibiMMDDetector"

    data_id = "bow_50"
    detector = AlibiMMDDetector(backend = 'pytorch')
    default_fit   (detector_id, detector, data_id, data[data_id]['orig'][0],    results, False)
    default_detect(detector_id, detector, data_id, data[data_id]['drifted'][0], results, False)

    save_results(results_file, results)

    data_id = "bow_768"
    detector = AlibiMMDDetector(backend = 'pytorch')
    default_fit   (detector_id, detector, data_id, data[data_id]['orig'][0],    results, False)
    default_detect(detector_id, detector, data_id, data[data_id]['drifted'][0], results, False)

    save_results(results_file, results)
    
    
    
if(False): 
    from detectors.CosineDetector import CosineSimilarityDriftDetector
    detector_id = "CosineDetector"

    data_id = "bow_50"
    detector = CosineSimilarityDriftDetector()
    default_fit   (detector_id, detector, data_id, data[data_id]['orig'][0],    results, False)
    default_detect(detector_id, detector, data_id, data[data_id]['drifted'][0], results, False)

    save_results(results_file, results)

    data_id = "bow_768"
    detector = CosineSimilarityDriftDetector()
    default_fit   (detector_id, detector, data_id, data[data_id]['orig'][0],    results, False)
    default_detect(detector_id, detector, data_id, data[data_id]['drifted'][0], results, False)

    save_results(results_file, results)



if(True): 
    from detectors.FCITDetector import FCITDriftDetector
    detector_id = "FCITDetector"

    if(False): 
        data_id = "bow_50"
        detector = FCITDriftDetector()
        default_fit   (detector_id, detector, data_id, data[data_id]['orig'][0],    results, False)
        default_detect(detector_id, detector, data_id, data[data_id]['drifted'][0], results, False)

        save_results(results_file, results)

    # Canceled after 2:15 without any result
    data_id = "bow_768"
    detector = FCITDriftDetector()
    default_fit   (detector_id, detector, data_id, data[data_id]['orig'][0],    results, False)
    default_detect(detector_id, detector, data_id, data[data_id]['drifted'][0], results, False)
    
    save_results(results_file, results)
