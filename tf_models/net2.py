import numpy as np
import tensorflow as tf
import netTools
import time
import math
#import scipy.io
import h5py

"""                                    OLD MATLAB MEHOD
PICKLE_FILE = 'gt5_20k.pickle'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
TENSORBOARD_DIR = 'tensorBoard'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
SAVE_DIR = 'saveDir/'

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
IMAGE_SIZE = 128
TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

BATCH_SIZE = 100
TOTAL_STEPS = 500001
HIDDEN_UNITS = 16
LEARNING_RATE = 0.5

# Load and separate datasets
data = netTools.load_pickle_dataset(PICKLE_FILE)
train_dataset, train_labels = data[TRAIN_INDEX]
valid_dataset, valid_labels = data[VALID_INDEX]
test_dataset, test_labels = data[TEST_INDEX]
"""


#DATA_FILE = 'arbAng_gt5.mat'
N2_DATA_FILE = '8ang1_30ms_midActv.mat'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
#TENSORBOARD_DIR = 'net4_arbang25m/'
N2_TENSORBOARD_DIR = 'net1_arbang/'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
N2_SAVE_DIR = N2_TENSORBOARD_DIR
N2_MODEL_ID = 'net1_arbang.ckpt'
N2_IMG_DIR = N2_TENSORBOARD_DIR + ''
N2_LOAD_MODEL = False
N2_SAVE_MODEL = True
N2_WRITE_IMAGES = True

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
#IMAGE_SIZE = 128
#TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Hyper params
N2_BATCH_SIZE = 100
N2_TOTAL_STEPS = 1
N2_LEARNING_RATE = 0.5
num_hidden = 2


def runNet(datafile=N2_DATA_FILE, tensorboard_dir=N2_TENSORBOARD_DIR, save_dir=N2_SAVE_DIR,
            model_id=N2_MODEL_ID, image_dir=N2_IMG_DIR, load_model=N2_LOAD_MODEL,
            save_model=N2_SAVE_MODEL, write_image=N2_WRITE_IMAGES, batch_size=N2_BATCH_SIZE,
            total_steps=N2_TOTAL_STEPS, learning_rate=N2_LEARNING_RATE, other_params={}):

    # Load and separate datasets
    #data = scipy.io.loadmat(datafile)
    data = h5py.File(datafile)
    train_dataset = np.transpose(data.get('train_inputs'))
    train_labels = np.transpose(data.get('train_labels'))
    valid_dataset = np.transpose(data.get('valid_inputs'))
    valid_labels = np.transpose(data.get('valid_labels'))
    test_dataset = np.transpose(data.get('test_inputs'))
    test_labels = np.transpose(data.get('test_labels'))


    #Get input size
    kx = int(data.get('kx')[0])
    ky = int(data.get('ky')[0])
    assert kx == ky
    # NETWORK PARAMS
    image_size = kx
    num_inputs = image_size * image_size
    num_outputs = image_size * image_size

    # Build graph
    # Using the standard graph
    with tf.Graph().as_default():
        
        # Network inputs and outputs
        input_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            num_inputs))
        label_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            num_outputs))
        
        # Define hidden layer
        with tf.name_scope('hidden'):
            weights = tf.Variable(tf.truncated_normal([num_inputs, num_hidden], 
                                    stddev=1.0 / num_hidden),
                                    name='hid_weights')
        
            biases = tf.Variable(tf.truncated_normal([num_hidden], 
                                    stddev=1.0 / num_hidden),
                                    name='hid_biases')
            hidden = tf.nn.relu(tf.matmul(input_placeholder, weights) + biases)

        #tf.image_summary("hid_weights", weights.reshape([1, TOTAL_PIXELS, HIDDEN_UNITS, 1]))
        tf.histogram_summary("hid_biases", biases)
        tf.histogram_summary("hid_weights", weights)
        tf.histogram_summary("hidden", hidden)

        # Define output layer
        with tf.name_scope('out_layer'):
            weights = tf.Variable(
                tf.truncated_normal([num_hidden, num_outputs],
                                    stddev=1.0 / num_hidden),
                name='out_weights')

            biases = tf.Variable(tf.truncated_normal([num_outputs], 
                                    stddev=1.0 / num_hidden),
                                    name='out_biases')
            logits = tf.matmul(hidden, weights) + biases

        tf.histogram_summary("out_biases", biases)
        tf.histogram_summary("out_weights", weights)
        tf.histogram_summary("logits", logits)

        # loss function - Sum squared difference (well mean...)
        loss = tf.div(tf.reduce_sum(
            tf.square(label_placeholder - logits)), 
            num_outputs*batch_size, name='loss')
        #loss = tf.reduce_sum(tf.square(label_placeholder - logits), name='loss')
        """

        # Linear Loss - sum( (t - a)^2 * lamb(t) )
        with tf.name_scope('loss_layer'):
            with tf.name_scope('lamb'):
                # Computer L(actual) = actual * m + c
                g = float(10) # Grayness (or whiteness) of the scene
                m = tf.constant( (num_outputs - (2.0 * g)) / num_outputs, dtype=tf.float32 )
                c = tf.constant( float(g) / num_outputs, dtype=tf.float32)
                lamb = tf.mul(m, logits) + c

            squr = tf.square(label_placeholder - logits, name='squr')
            tmp = tf.mul(squr, lamb, name='tmp')
            loss = tf.reduce_mean(tf.div(tmp + 1e-9, num_outputs), name='loss')

        tf.histogram_summary('tmp', tmp)
        tf.histogram_summary('lamb', lamb)
        tf.histogram_summary('squr', squr)
        """
        
        # Partwise loss - sum(relu(l - p) * ~1) + sum(relu(p - l) * ~0)
        # If it doesn't guess a white highenough its a big loss but if it mislabels
        # a 0, its not a big deal
        """
        with tf.name_scope('loss_layer'):
            g = float(10)
            wc = tf.constant( (TOTAL_PIXELS - g) / float(TOTAL_PIXELS) )
            bc = tf.constant( g / TOTAL_PIXELS )
            ww = tf.reduce_sum(tf.mul(tf.nn.relu(tf.sub(label_placeholder, logits)), wc)) # Wrong white
            wb = tf.reduce_sum(tf.mul(tf.nn.relu(tf.sub(logits, label_placeholder)), bc)) # wrong black
            loss = tf.add(ww, wb, name='loss')

        tf.histogram_summary('ww', ww)
        tf.histogram_summary('wb', wb)
        """

        # Log data
        tf.scalar_summary(loss.op.name, loss)

        # Training operations
        optimiser = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimiser.minimize(loss, global_step=global_step)
        
        # Collect all summaries together into one operation
        summary_op = tf.merge_all_summaries()
        
        # Saver will save the state of the network in case of crash etc.
        saver = tf.train.Saver()


        # Run graph
        sess = tf.Session()
        
        # Initialise variables 
        init = tf.initialize_all_variables()
        sess.run(init)

        if tf.__version__ == '0.7.0':  # Bug with tf version on cluster
            summary_writer = tf.train.SummaryWriter(tensorboard_dir)
        else: 
            summary_writer = tf.train.SummaryWriter(tensorboard_dir, 
                                                    graph_def=sess.graph_def)
        
        for step in range(total_steps):
            start_time = time.time()

            # Create feed dictionary
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset : (offset + batch_size), :]
            batch_labels = train_labels[offset : (offset + batch_size), :]
            feed_dict = { input_placeholder : batch_data, 
                        label_placeholder : batch_labels 
                        }

            """
            ## DEBUGGING ################
            # Just compute the loss
            loss_value = sess.run([loss], feed_dict=feed_dict)
            # THen write the loss and other stats to file
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

            continue  # Force loop restart
            #################    END DEBUG   ################
            """

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 250 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                
            # Save a checkpoint and evaluate the model periodically.
            if step % total_steps - 1 == 0:
               eval_model(step, batch_size, valid_labels, 
                          valid_dataset, sess, logits, loss, input_placeholder, label_placeholder)     


    if save_model:
        save_path = saver.save(sess, save_dir + model_id)
        print("Model saved in file: %s" % save_path)


    ## If on my laptop then just write straight to images
    if write_image:
        print("Attempting to write images now...")
        import visTools

        preds = eval_model(step, batch_size, valid_labels, 
                          valid_dataset, sess, logits, loss, input_placeholder, label_placeholder)
        visTools.write_preds(batch_data, batch_labels, preds, image_dir, kx)

def eval_model(step, batch_size, valid_labels, valid_dataset, sess, logits, loss, input_placeholder, label_placeholder):
    # RUN Network with validation data
    offset = (step * batch_size) % (valid_labels.shape[0] - batch_size)
    batch_data = valid_dataset[offset : (offset + batch_size), :]
    batch_labels = valid_labels[offset : (offset + batch_size), :]
    feed_dict = { input_placeholder : batch_data, 
              label_placeholder : batch_labels 
              }

    preds, validation_value, = sess.run([logits, loss], feed_dict=feed_dict)
    print("Step: %d: validation: %.5f" % (step, validation_value))
    return preds




