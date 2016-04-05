import numpy as np
import tensorflow as tf
import netTools
import time

PICKLE_FILE = 'gt5_20k.pickle'
TENSORBOARD_DIR = '/home/Students/s4290365/thesis/tf_models/tensorBoard'

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2

BATCH_SIZE = 100
TOTAL_STEPS = 101
IMAGE_SIZE = 128
TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE
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
    label_placeholder = tf.placeholder(tf.float3s, shape=(BATCH_SIZE,
                                                        TOTAL_PIXELS))
    
    # Define hidden layer
    with tf.name_scope('hidden'):
        weights = tf.Variable(
            tf.truncated_normal([TOTAL_PIXELS, HIDDEN_UNITS], 
                                stddev=1.0 / math.sqart(float(TOTAL_PIXELS)),
            name='weights')
    
        biases = tf.Variable(tf.zeros([HIDDEN_UNITS]),
                                name='biases')
        hidden = tf.nn.relu(tf.matmul(input_placeholder, weights) + biases)

    # Define output layer
    with tf.name_scope('out_layer'):
        weights = tf.Variable(
            tf.truncated_normal([HIDDEN_UNITS, TOTAL_PIXELS],
                                stddev=1.0 / math.sqrt(float(TOTAL_PIXELS)),
            name='weights')

        biases = tf.Variable(tf.zeros([TOTAL_PIXELS]),
                                name='biases')    
        logits = tf.matmul(hidden, weights) + biases

    # Could consider using reduce_sum and normalise
    # loss function - Sum squared difference (well mean...)
    ssd = tf.reduce_mean(tf.square(label_placeholder - logits), name='ssd')
    tf.scalar_summary(ssd.op.name, ssd)

    # Training operations
    optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimiser.minimize(ssd, global_step=global_step)
    
    # Collect all summaries together into one operation
    #summary_op = tf.merge_all_summaries()
    
    # Saver will save the state of the network in case of crash etc.
    saver = tf.train.Saver()


    # Run graph
    sess = tf.Session()
    
    # Initialise variables 
    init = tf.initalize_all_variables()
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
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file. - Not used atm
            #summary_str = sess.run(summary_op, feed_dict=feed_dict)
            #summary_writer.add_summary(summary_str, step)









