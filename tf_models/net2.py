import numpy as np
import tensorflow as tf
import netTools
import time
import math

PICKLE_FILE = 'gt5_20k.pickle'
TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
IMAGE_SIZE = 128
TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

BATCH_SIZE = 100
TOTAL_STEPS = 4001
HIDDEN_UNITS = 256
LEARNING_RATE = 0.01

# Load and separate datasets
data = netTools.load_pickle_dataset(PICKLE_FILE)
train_dataset, train_labels = data[TRAIN_INDEX]
valid_dataset, valid_labels = data[VALID_INDEX]
test_dataset, test_labels = data[TEST_INDEX]

# Build graph
# Using the standard graph
with tf.Graph().as_default():
    
    # Network inputs and outputs
    input_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,
                                                        TOTAL_PIXELS))
    label_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,
                                                        TOTAL_PIXELS))
    
    # Define hidden layer
    with tf.name_scope('hidden'):
        weights = tf.Variable(
            tf.truncated_normal([TOTAL_PIXELS, HIDDEN_UNITS], 
                                stddev=1.0 / math.sqrt(float(TOTAL_PIXELS)),
            name='weights'))
    
        biases = tf.Variable(tf.zeros([HIDDEN_UNITS]),
                                name='biases')
        hidden = tf.nn.relu(tf.matmul(input_placeholder, weights) + biases)

    tf.histogram_summary("hid_weights", weights)
    tf.histogram_summary("hid_biases", biases)

    # Define output layer
    with tf.name_scope('out_layer'):
        weights = tf.Variable(
            tf.truncated_normal([HIDDEN_UNITS, TOTAL_PIXELS],
                                stddev=1.0 / math.sqrt(float(TOTAL_PIXELS)),
            name='weights'))

        biases = tf.Variable(tf.zeros([TOTAL_PIXELS]),
                                name='biases')    
        logits = tf.matmul(hidden, weights) + biases

    # Could consider using reduce_sum and normalise
    # loss function - Sum squared difference (well mean...)
    loss = tf.div(tf.reduce_sum(tf.square(label_placeholder - logits), name='loss'), TOTAL_PIXELS*BATCH_SIZE)
    #loss = tf.reduce_sum(tf.square(label_placeholder - logits), name='loss')

    # Log data
    tf.image_summary("prediction", logits)
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
    init = tf.initialize_all_variables()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(TENSORBOARD_DIR,
                                            graph_def=sess.graph_def)
    
    for step in xrange(TOTAL_STEPS):
        start_time = time.time()

        # Create feed dictionary
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset : (offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset : (offset + BATCH_SIZE), :]
        feed_dict = { input_placeholder : batch_data, 
                    label_placeholder : batch_labels 
                    }

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
    
        duration = time.time() - start_time

        if step % 100 == 0:
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            
            # Save a checkpoint and evaluate the model periodically.
            if step % 1000 == 0:
                saver.save(sess, SAVE_DIR, global_step=step)
                
                # RUN Network with validation data
                offset = (step * BATCH_SIZE) % (valid_labels.shape[0] - BATCH_SIZE)
                batch_data = valid_dataset[offset : (offset + BATCH_SIZE), :]
                batch_labels = valid_labels[offset : (offset + BATCH_SIZE), :]
                feed_dict = { input_placeholder : batch_data, 
                            label_placeholder : batch_labels 
                            }

                validation_value = sess.run(loss, feed_dict=feed_dict)
                print("Step: %d: validation: %.5f" % (step, validation_value))






