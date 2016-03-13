""" A simple neural net used to ensure all hardware and software is working to
load in and process some test data. 
"""
# Try loading some useful libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import tarfile
import urllib
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import cPickle as pickle

image_size = 128
pixel_depth = 255.0  # TODO double check this is correct

def load(data_folder, max_num_images):
    """ Load in all images as a numpy array in the given data folder

    Precondition: data_folder must be a valid path and contain images

    """
    dataset = np.ndarray(
        shape=(max_num_images, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(
        shape=(max_num_images, image_size, image_size), dtype=np.float32)
    image_index = 0
    for image in os.listdir(data_folder):
        if 'futr' in image: # Skip futr images
            continue;

        if image_index >= max_num_images:
            raise Exception('More images than expected')
        #image_number = image[24:29] # TODO This is a hardcode to get number...
        
        image_file = os.path.join(data_folder, image)
        label_file = image_file.replace('past', 'futr')
        
        try:
            image_data = (ndimage.imread(image_file).astype(float) - 
                            pixel_depth / 2) / pixel_depth
            label_data = (ndimage.imread(label_file).astype(float) - 
                            pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size) or label_data.shape != (image_size, image_size):
                raise Exception("Image not in correct format - Skipping")
            dataset[image_index, :, :] = image_data
            labels[image_index, :, :] = label_data
            image_index += 1
        except IOError as e:
            print "Could not read: " + image_file + " - skipped"

    num_images = image_index
    dataset = dataset[0:num_images]
    labels = labels[0:num_images]
    print num_images
    print 'Full dataset tensor: ' + str(dataset.shape)
    print 'Mean: ' + str(np.mean(dataset))
    print 'Standard deviation: ' + str(np.std(dataset))
    print 'Label tensor shape: ' + str(labels.shape)
    return dataset, labels

dataset, labels = load('relearn/data', 3000)

#plt.imshow(dataset[1, :, :], plt.cm.gray)
#plt.show()  # Verify data

np.random.seed(133)  # Random but consistant ;)
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation, :, :]
    return shuffled_dataset, shuffled_labels

dataset, labels = randomize(dataset, labels)

# Split into training, validation and test sets
num_train = int(labels.shape[0] * 0.7) # 70% is training data
num_test = int(labels.shape[0] * 0.9) # 20% is validation, last 10% is test

train_dataset = dataset[0:num_train, :, :]
train_labels = labels[0:num_train, :, :]
valid_dataset = dataset[num_train:num_test, :, :]
valid_labels = labels[num_train:num_test, :, :]
test_dataset = dataset[num_test:, :, :]
test_labels = labels[num_test:, :, :]

print "-" * 7 + "FINAL SIZES" + "-" * 7
print "Train dataset:", train_dataset.shape
print "Train labels:", train_labels.shape
print "valid dataset:", valid_dataset.shape
print "valid labels:", valid_labels.shape
print "test dataset:", test_dataset.shape
print "test labels:", test_labels.shape


# Save as a pickle for easy reading later on
pickle_file = 'relearn.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    raise Exception("Problem saving pickle file: " + str(e))
    
print "Compressed pickle size: " + str(os.stat(pickle_file))

