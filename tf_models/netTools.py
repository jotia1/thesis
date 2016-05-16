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


def save_weights(weights, outfile):
    """ Given a tensors weights as a numpy array save to the outfile as an npz
    """
    np.savez(outfile, weights=weights)
    print("Saved: " + outfile)

def save_preds(intensor, predictions, filename, batch_size=100):
    """ Given network input and predicton tensors save them to given file
        Tensor should be in the form [BATCHSIZE, TOTAL_PIXELS]
    """
    np.savez(filename, ins=intensor, preds=predictions)
    print("Saved", filename)

