import numpy as np
import tensorflow as tf
import netTools
import time
import math
import scipy.io

#DATA_FILE = 'arbAng_gt5.mat'
DATA_FILE = '8ang1_30ms_midActv.mat'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
TENSORBOARD_DIR = 'autoEnc/'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
SAVE_DIR = TENSORBOARD_DIR
MODEL_ID = 'autoEnc.ckpt'
IMG_DIR = TENSORBOARD_DIR + 'imgs/'
LOAD_MODEL = True
SAVE_MODEL = False
WRITE_IMAGES = True

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
#IMAGE_SIZE = 128
#TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Hyper params
BATCH_SIZE = 100
TOTAL_STEPS = 1
LEARNING_RATE = 0.1

# Load and separate datasets
data = scipy.io.loadmat(DATA_FILE)
train_dataset = data.get('train_inputs')
train_labels = data.get('train_labels')
valid_dataset = data.get('valid_inputs')
valid_labels = data.get('valid_labels')
test_dataset = data.get('test_inputs')

#Get input size
kx = int(data.get('kx'))
ky = int(data.get('ky'))

# NETWORK PARAMS
NUM_INPUT_UNITS = kx * ky
#HIDDEN_UNITS = 16
NUM_OUTPUT_UNITS = kx * ky
# Hidden layer
NUM_HIDDEN = 121

print("INPUT_UNITS:", NUM_INPUT_UNITS, "OUTPUT_UNITS:", NUM_OUTPUT_UNITS)


# Build graph
# Using the standard graph
with tf.Graph().as_default():
    
    # Network inputs and outputs
    input_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,
                                                        NUM_INPUT_UNITS))
    label_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,
                                                        NUM_OUTPUT_UNITS))
    
    # Hidden layer
    with tf.name_scope('hidden'):
        weights = tf.Variable(
            tf.truncated_normal([NUM_INPUT_UNITS, NUM_HIDDEN],
                                stddev=0.1),
            name='weights')

        biases = tf.Variable(tf.truncated_normal([NUM_HIDDEN], 
                                stddev=0.1),
            name='biases')
        hidden = tf.nn.sigmoid(tf.matmul(input_placeholder, weights) + biases, name='hidden')
        

    tf.histogram_summary("hid_out_biases", biases)
    tf.histogram_summary("hid_out_weights", weights)
    tf.histogram_summary("Hid_output", hidden)

    # Define output layer
    with tf.name_scope('out_layer'):
        weights = tf.Variable(
            tf.truncated_normal([NUM_HIDDEN, NUM_OUTPUT_UNITS],
                                stddev=0.1),
            name='weights')

        biases = tf.Variable(tf.truncated_normal([NUM_OUTPUT_UNITS], 
                                stddev=0.1),
            name='biases')
        logits = tf.matmul(hidden, weights) + biases

    tf.histogram_summary("out_biases", biases)
    tf.histogram_summary("out_weights", weights)
    tf.histogram_summary("logits", logits)

    # loss function - Sum squared difference (well mean...)
    loss = tf.reduce_mean(
        tf.square(label_placeholder - logits), name='loss')


    # Log data
    tf.scalar_summary(loss.op.name, loss)

    # Training operations
    optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimiser.minimize(loss, global_step=global_step)
    
    # Collect all summaries together into one operation
    summary_op = tf.merge_all_summaries()
    
    # Saver will save the state of the network in case of crash etc.
    saver = tf.train.Saver()


    # Run graph
    sess = tf.Session()
    
    # Initialise variables 
    if not LOAD_MODEL:
        init = tf.initialize_all_variables()
        sess.run(init)
    else:
        saver.restore(sess, SAVE_DIR + MODEL_ID)
        print("Restored network from file: %s" % SAVE_DIR + MODEL_ID)

    summary_writer = tf.train.SummaryWriter(TENSORBOARD_DIR,
                                            graph_def=sess.graph_def)
    
    for step in xrange(TOTAL_STEPS):
        # Create feed dictionary
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset : (offset + BATCH_SIZE), :]
        #batch_labels = train_labels[offset : (offset + BATCH_SIZE), :]
        feed_dict = { input_placeholder : batch_data, 
                    label_placeholder : batch_data 
                    }

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        if True: #step % 25 == 0:
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            
            # RUN Network with validation data
            offset = (step * BATCH_SIZE) % (valid_labels.shape[0] - BATCH_SIZE)
            batch_data = test_dataset[offset : (offset + BATCH_SIZE), :]
            #batch_labels = valid_labels[offset : (offset + BATCH_SIZE), :]
            feed_dict = { input_placeholder : batch_data, 
                        label_placeholder : batch_data 
                        }

            preds, validation_value, = sess.run([logits, loss], feed_dict=feed_dict)
            if True: #step % 100 == 0:
                print("Step: %d: validation: %.5f" % (step, validation_value))

            if True: #WRITE_IMAGES and step == TOTAL_STEPS:
                print("Attempting to write images now...")
                import visTools

                visTools.write_preds(batch_data, batch_data, preds, IMG_DIR, kx)
                

if SAVE_MODEL:
    save_path = saver.save(sess, SAVE_DIR + MODEL_ID)
    print("Model saved in file: %s" % save_path)


## If on my laptop then just write straight to images











