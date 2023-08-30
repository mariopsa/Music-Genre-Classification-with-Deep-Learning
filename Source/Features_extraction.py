import os           # access to folder and subfolder
import librosa, librosa.feature
import math
import json        #  export json file

    # Constants
DATASET_PATH = "dataset"         #dataset input
JSON_PATH = "data_test.json"         #output json file path
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

#function to save coefs and label the data in arrays
def extract_data(dataset_path, json_path, n_mfcc = 13, n_fft=2048, hop_lenght=512, num_segments=10):

    #build a dictionary to store data    
    data = {
        
        "mapping" : [],
        "mfcc" : [],
        "labels" : [],
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    
    #We need an uniform number of mfcc vectors we round to an integer number
    expected_num_mfcc_vector_per_segment = math.ceil(num_samples_per_segment/hop_lenght) 
    
    #loop for all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)): 
        
        # ensure that we´re not at the root level
        if dirpath is not dataset_path:
    
            #save the semantic label
            dirpath_components = dirpath.split("/")     # genre/blues => ["genre","blues"]
            semantic_label = dirpath_components[-1]     # take the last component
            data["genre"].append(semantic_label)
            
            print("\n Processing {}".format(semantic_label))
        
        # process files for a specific genre
        for f in filenames:

            if f == ".DS_Store":
                continue            
            
            #load the audio file
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            #process segments extracting mfcc and storing data
            for s in range(num_segments):                           # s is the index for current segment
                
                #Start and finish sample of the segments
                start_sample = num_samples_per_segment * s          
                finish_sample = start_sample + num_samples_per_segment
                
                #Extract mfccs with the librosa function
                mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                            sr=sr,
                                            n_fft = n_fft,
                                            n_mfcc=n_mfcc,
                                            hop_lenght=hop_lenght)
                
                mfcc = mfcc.T
                               
                #store mfcc for segment if it has the expected length
                if len(mfcc) == expected_num_mfcc_vector_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1) #first iteration is for the data set path
                    print("{}, segment:{}".format(file_path, s+1))

    #Open the json file to read
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    
#Script execute
if __name__ == "main" :  
  extract_data(DATASET_PATH, JSON_PATH, num_segments=10)
    
    
    
    
    
    
    
    
    











