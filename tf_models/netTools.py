import cPickle as pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


    #netin = tensor2img(intensor, batch_size)
    #preds = tensor2img(predictions, batch_size)
    #f = open(picklename, 'wb')
    #save = {
    #    'netin': netin,
    #    'preds': preds,
    #}
    #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    #f.close()
    #np.save(filename, save)
    print("Saved", filename)

def show_preds(filename):
    """ Open a pickle file containing predictions and display them asking the
        user for input
    """
    dataset = np.load(filename)
    if 'ins' not in dataset.files or 'preds' not in dataset.files:
        raise Exception("Dataset missing ins or preds, may not be valid")
    ins = dataset['ins']
    preds = dataset['preds']
    assert ins.shape == preds.shape
    
    raise Exception("Function NOT Finished")

    cur_img = 0        

    while True:
        usr = input("Action (n/b/q):")
        if usr == 'n':
            pass
        elif usr == 'b':
            pass
        elif usr == 'q':
            break
        else:
            print("Unreccognised command:", usr)

def preds2png(filename, outdir):
    """ Given a saved file convert in/prediction pairs to pngs and save to
        outdir
        Preconditon: Assumes outdir exists and is writeable
                    filename is valid and readable
    """
     
    dataset = np.load(filename)
    if 'ins' not in dataset.files or 'preds' not in dataset.files:
        raise Exception("Dataset missing ins or preds, may not be valid")
    ins = dataset['ins']
    preds = dataset['preds']
    del dataset
    assert ins.shape == preds.shape
    
    batch_size = ins.shape[0]
    
    for i in xrange(batch_size):
        in_img = ins[i, :].reshape(IMAGE_SIZE, IMAGE_SIZE)
        pred_img = preds[i, :].reshape(IMAGE_SIZE, IMAGE_SIZE)

        plt.imshow(in_img, cmap='gray')  # Save input img
        plt.savefig(outdir + str(i) + "in.png")
    
        plt.imshow(pred_img, cmap='gray')  # save prediction
        plt.savefig(outdir + str(i) + "out.png")
        
        # When visualising weights can use something like
        # http://stackoverflow.com/questions/11775354/how-can-i-display-a-np-array-with-pylab-imshow
        # im = plt.imshow(arr, cmap='hot')
        # plt.colorbar(im, orientation='horizontal')
        # plt.show()        
        
        # multiple images in one figure
        # http://stackoverflow.com/questions/17111525/how-to-show-multiple-images-in-one-figure        









 
