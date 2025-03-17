import os.path
import sys
import h5py
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
import os
import fnmatch
 

##################################################
input_size = 150
 
data_folder = "../traces/puf0_avg100/"
model_folder = "../models/puf0/"
modelName = 'puf0.h5'

#data_folder = "../traces/puf12_avg100/"
#model_folder = "../models/puf12/"
#modelName = 'puf12'											#

#data_folder = "../traces/puf0_avg100/"
#model_folder = "../models/puf0/"
#modelName = 'puf0_avg0_blocks_of_1000'
################################################### 

def load_sca_model(model_folder,model_file):
    for f_name in os.listdir(model_folder):
        if f_name.startswith(model_file):
            model = load_model(model_folder+f_name)
            print(f_name)
            #print(model.summary())
        else:
            continue
    return model

def rank_func(model, dataset, labels):

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    if len(input_layer_shape) == 2:
        # This is a MLP
        input_data = dataset
    elif len(input_layer_shape) == 3:
        # This is a CNN: reshape the data
        input_data = dataset
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)

    # Predict probabilities for 2-node output case
    predictions = model.predict(input_data)
    predictions = np.argmax(predictions,axis=1)

    # xor = 0 if the same
    ranks_prd = np.bitwise_xor(predictions, labels)

    return (ranks_prd)


# Load testing data
def load_traces(repetitions):
    # Load traces and labels for the specified number of repetitions
    traces = np.load(data_folder+'trace.npy')[:repetitions,:]  # [0:1000,:]
    labels = np.load(data_folder+'label.npy')[:repetitions]  # [0:1000]
    print('traces shape:', traces.shape)
    print('labels shape:', labels.shape)

    important_trace_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 85, 86, 87, 88, 89, 90, 91, 92, 127, 128, 129, 130, 131, 132]
    traces = traces[:, important_trace_points]  # Select only these 25 points

    print('traces shape after selection:', traces.shape)

    # Scale (standardize) traces
    delimitedTraces = np.zeros(traces.shape)
    for x_index in range(traces.shape[0]):
        delimitedTraces[x_index,:] = -1 + (traces[x_index,:] - np.min(traces[x_index,:])) * 2 / (np.max(traces[x_index,:]) - np.min(traces[x_index,:]))

    return (delimitedTraces, labels)

# Check a saved model against test traces
def check_model(model_folder, model, count):
    # Load model
    model_load = load_sca_model(model_folder, model)
    all_predictions = []
    all_true_labels = []

    # Load traces and labels for the current repetition
    traces, labels = load_traces(1000)

    # Create a new trace with shape (0, 150) for the average trace
    #new_trace = np.zeros((0, 25))

    # Compute the average trace
    #average_trace = np.mean(traces, axis=0)
    #new_trace = np.vstack((average_trace, new_trace))

    new_trace = traces

    # Correct if 0
    num_attack = int(labels.shape[0])
    ranks_prd = np.zeros((num_attack, 1))

    # Change the trace depending on the strategy
    ranks_prd = rank_func(model_load, new_trace, labels)

    # Calculate classification accuracy for the current repetition
    classification_accuracy = 1 - np.mean(ranks_prd)

    # Print the classification accuracy for the average trace
    print(classification_accuracy)

    # Append classification accuracy and true labels
    # Used to get the sum of log
    all_predictions.append(classification_accuracy)
    all_true_labels.append(np.mean(labels))

    # Combine predictions using sum of logs
    predictions = np.sum(np.log(all_predictions), axis=0)

    # Calculate probability of correct prediction
    a = np.exp(predictions)
    print('test accuracy:', a)
    print('mean of labels:', all_true_labels)

    return
    
start = time.time()
print('input size =', input_size)
                            
# Test all models in model_folder
count = 0
for file_name in os.listdir(model_folder):
    if fnmatch.fnmatch(file_name, '*.h5'):  # Check that file is a model
        model = file_name
        count = count + 1
        check_model(model_folder,model,count)

end = time.time()
print("The total running time was: ",((end-start)/60), " minutes.") 

