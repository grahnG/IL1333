import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, InputLayer, BatchNormalization, ReLU, Softmax, Dropout
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
import time

################################################
data_folder = "../traces/puf0_avg0_blocks_of_1000/"
model_folder = "../models/puf0/"
modelName = 'puf0_avg0_blocks_of_1000'

input_size = 25
#################################################

def create_model(classes=2, input_size=25):
    input_shape = (25,)

    # Create model.
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Dense(4, kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.5))  # Add dropout for regularization

    model.add(Dense(classes, kernel_initializer='he_uniform', input_dim=32))
    model.add(Softmax())

    optimizer = Nadam(lr=0.001, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def load_traces():
    # Import our traces
    traces = np.load(data_folder + 'trace.npy')
    labels = np.load(data_folder + 'label.npy')
    for i in range(1, 1):
        tempTrace = np.load(data_folder + 'trace' + str(i) + '.npy')
        tempLabel = np.load(data_folder + 'label' + str(i) + '.npy')
        traces = np.append(traces, tempTrace, axis=0)
        labels = np.append(labels, tempLabel, axis=0)

    print('traces shape:', traces.shape)
    print('labels shape:', labels.shape)

    # Scale (standardize) traces
    important_trace_points = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 85, 86, 87, 88, 89, 90, 91, 92, 127, 128, 129, 130, 131]
    traces = traces[:, important_trace_points]

    delimitedTraces = np.zeros(traces.shape)
    for x_index in range(traces.shape[0]):
        delimitedTraces[x_index, :] = -1 + (traces[x_index, :] - np.min(traces[x_index, :])) * 2 / (np.max(traces[x_index, :]) - np.min(traces[x_index, :]))

    # Shuffle the data before splitting
    indices = np.arange(len(delimitedTraces))
    np.random.shuffle(indices)
    delimitedTraces = delimitedTraces[indices]
    labels = labels[indices]

    # Sequential split (last 10% as test set)
    num_samples = len(delimitedTraces)
    split_index = int(0.9 * num_samples)

    X_train, X_test = delimitedTraces[:split_index], delimitedTraces[split_index:]
    Y_train, Y_test = labels[:split_index], labels[split_index:]

    print(f"Dataset split: {split_index} training samples, {num_samples - split_index} testing samples")

    return X_train, X_test, Y_train, Y_test

def train_model(X_profiling, Y_profiling, model, save_file_name, epochs=50, batch_size=64):
    check_file_exists(os.path.dirname(save_file_name))
    save_model = ModelCheckpoint(save_file_name + '.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    input_layer_shape = model.get_layer(index=0).input_shape

    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)

    if len(input_layer_shape) == 2:
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=20)

    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=2), batch_size=batch_size, verbose=0, epochs=epochs, callbacks=[es, save_model], validation_split=0.3)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../history/puf0/' + modelName + '.pdf')
    plt.show()

    return history

# Start of execution
start = time.time()

# Load the training traces
X_train, X_test, Y_train, Y_test = load_traces()

# MLP training
mlp = create_model(input_size=input_size)
train_model(X_train, Y_train, mlp, model_folder + modelName, epochs=100, batch_size=128)

end = time.time()
print("The total running time was: ", ((end - start) / 60), " minutes.")

# Evaluate on the full test set
test_loss, test_acc = mlp.evaluate(X_test, to_categorical(Y_test, num_classes=2))
print("Test Accuracy:", test_acc)

# Evaluate on a small subset (e.g., 10 samples)
random_indices = np.random.choice(len(X_test), 10, replace=False)
X_sample = X_test[random_indices]
Y_sample = Y_test[random_indices]
sample_loss, sample_acc = mlp.evaluate(X_sample, to_categorical(Y_sample, num_classes=2))
print("Sample Accuracy:", sample_acc)

