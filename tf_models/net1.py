import numpy as np
import tensorflow as tf
import scipy.io

"""                                            OLD MATLAB METHOD
pickle_file = 'gt5_20k.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print 'Training set', train_dataset.shape, train_labels.shape
    print 'Validation set', valid_dataset.shape, valid_labels.shape
    print 'Test set', test_dataset.shape, test_labels.shape

image_size = 128
num_outputs = image_size * image_size  # output size

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print 'Training set', train_dataset.shape, train_labels.shape
print 'Validation set', valid_dataset.shape, valid_labels.shape
print 'Test set', test_dataset.shape, test_labels.shape
"""

################   DEFINING NETWORK #########################

#DATA_FILE = 'arbAng_gt5.mat'
N1_DATA_FILE = '8ang1_30ms_midActv.mat'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
#TENSORBOARD_DIR = 'net4_arbang25m/'
N1_TENSORBOARD_DIR = 'net1_arbang/'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
N1_SAVE_DIR = N1_TENSORBOARD_DIR
N1_MODEL_ID = 'net1_arbang.ckpt'
N1_IMG_DIR = N1_TENSORBOARD_DIR + ''
N1_LOAD_MODEL = False
N1_SAVE_MODEL = True
N1_WRITE_IMAGES = True

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
#IMAGE_SIZE = 128
#TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Hyper params
N1_BATCH_SIZE = 100
N1_TOTAL_STEPS = 1
N1_LEARNING_RATE = 0.5

def runNet(datafile=N1_DATA_FILE, tensorboard_dir=N1_TENSORBOARD_DIR, save_dir=N1_SAVE_DIR,
            model_id=N1_MODEL_ID, image_dir=N1_IMG_DIR, load_model=N1_LOAD_MODEL,
            save_model=N1_SAVE_MODEL, write_image=N1_WRITE_IMAGES, batch_size=N1_BATCH_SIZE,
            total_steps=N1_TOTAL_STEPS, learning_rate=N1_LEARNING_RATE):

    # Load and separate datasets
    data = scipy.io.loadmat(datafile)
    train_dataset = data.get('train_inputs')
    train_labels = data.get('train_labels')
    valid_dataset = data.get('valid_inputs')
    valid_labels = data.get('valid_labels')
    test_dataset = data.get('test_inputs')
    test_labels = data.get('test_labels')
    

    #Get input size
    kx = int(data.get('kx'))
    ky = int(data.get('ky'))
    assert kx == ky
    # NETWORK PARAMS
    image_size = kx
    num_inputs = image_size * image_size
    num_outputs = image_size * image_size

    graph = tf.Graph()

    with graph.as_default():
        
        # Load all the input data as constants on the graph
        tf_train_dataset = tf.placeholder(tf.float32,
                shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32,
                shape=(batch_size, image_size * image_size))
        tf_valid_dataset = tf.constant(valid_dataset, dtype=tf.float32)
        tf_test_dataset = tf.constant(test_dataset, dtype=tf.float32)

        # The actual weights to be trained. Initialise using random values from a
        # truncated normal distribution and biases at zero.
        weights = tf.Variable(tf.truncated_normal([image_size * image_size, \
                    num_outputs]))
        biases = tf.Variable(tf.zeros([num_outputs]))
        
        # Compuation 
        # Can replace sigmoid with tf.nn.relu
        hidden_output = tf.sigmoid(tf.matmul(tf_train_dataset, weights) + biases)
        # Hopefully this is Euclidean loss
        loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(hidden_output, tf_train_labels), 2)))

        # Optimizer - Gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        train_prediction = hidden_output
        valid_prediction = tf.matmul(tf_valid_dataset, weights) + biases
        test_prediction = tf.matmul(tf_test_dataset, weights) + biases

    ################   ACTUAL TRAINING  ###################

    def accuracy(pred, labels):
        return tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(pred, labels), 2))).eval()
        

    with tf.Session(graph=graph) as session:
        writer = tf.train.SummaryWriter(save_dir, graph_def=session.graph_def)
        writer.flush()
        writer.close()
        tf.initialize_all_variables().run()
        print "Initialized"

        for step in xrange(total_steps):
            print "step:", step
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Create minibatch
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, 
                          tf_train_labels : batch_labels}

            _, l, predictions = session.run([optimizer, loss, train_prediction], 
                feed_dict=feed_dict)
            
            if step % 100 == 0:
                print "loss at step", step, ":", l
                print type(valid_prediction.eval())
                print type(valid_labels)
                print "Training accuracy: %.1f%%" % accuracy(predictions, batch_labels)
                print 'Validation accuracy: %1.f%%' % accuracy(valid_prediction.eval(), valid_labels)
        print 'Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels)


if __name__ == '__main__':
    runNet1()
