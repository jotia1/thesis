import cPickle as pickle
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 128

def reformat(dataset, labels):
    """ Reformats a tensor to flatten the images
    """
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    labels = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    return dataset, labels

def tensor2img(ten, batch_size):
    """ Convert given tensor to a 4D tensor for an image_summary
    """
    return tf.reshape(ten, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1])


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

    num_outputs = IMAGE_SIZE * IMAGE_SIZE  # output size

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    return (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels)


def save_preds(intensor, predictions, filename, batch_size=100):
    """ Given network input and predicton tensors save them to given file
        Tensor should be in the form [BATCHSIZE, TOTAL_PIXELS]
    """
    np.savez(filename, ins=intensor, preds=predictions)
    print("Saved", filename)

