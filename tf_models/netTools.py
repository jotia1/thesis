import cPickle as pickle
import numpy as np
import tensorflow as tf


def reformat(dataset, labels):
    """ Reformats a tensor to flatten the images
    """
    image_size = 128
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    return dataset, labels

def load_pickle_dataset(pickleName):
    """ Load the given pickle file

    Preconditions: Assumes pickle file exists and is valid
        Assumes training, validation and test datasets
        Assumes data is images 128 pixels by 128 pixels

    Returns three tuple of datasets 
    """
    with open(pickleName, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    image_size = 128
    num_outputs = image_size * image_size  # output size

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    return (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels)

