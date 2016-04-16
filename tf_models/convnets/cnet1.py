import numpy as np
import tensorflow as tf
#import netTools
import time
import math
import scipy.io

DATA_FILE = '8ang1_30ms.mat'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
TENSORBOARD_DIR = 'tensorBoard'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
SAVE_DIR = 'saveDir/'

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
#IMAGE_SIZE = 128
#TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

BATCH_SIZE = 100
TOTAL_STEPS = 500001
#HIDDEN_UNITS = 16
LEARNING_RATE = 0.1

# CONVOLUTION params
# TODO dynamically congifure this? inc. in mat file?
KX = 11
KY = 11
KZ = 1  # Use depth 1 (gray scale (decayed) images)
TOTAL_VOXELS = KX * KY * KZ
NUM_FEATURES = 9
CONV_SIZE = 6
FC_UNITS = 1024  # TODO why?
NUM_INPUTS = TOTAL_VOXELS
NUM_OUTPUTS = TOTAL_VOXELS



# Load and separate datasets
"""
data = netTools.load_pickle_dataset(PICKLE_FILE)
train_dataset, train_labels = data[TRAIN_INDEX]
valid_dataset, valid_labels = data[VALID_INDEX]
test_dataset, test_labels = data[TEST_INDEX]
"""
data = scipy.io.loadmat(DATA_FILE)
train_dataset = data.get('dataset')
train_labels = data.get('labels')
vaid_dataset = train_dataset
valid_labels = train_labels # TODO just while getting network running....


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')


# Build graph
# Using the standard graph
with tf.Graph().as_default():
    
    # Network inputs and outputs
    # TODO worry about (x, y) position
    input_placeholder = tf.placeholder(tf.float32, 
                            shape=[BATCH_SIZE, NUM_INPUTS])
    label_placeholder = tf.placeholder(tf.float32, 
                            shape=[BATCH_SIZE, NUM_INPUTS])

    # Define first convolution
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([CONV_SIZE, CONV_SIZE, 1, NUM_FEATURES])
        b_conv1 = bias_variable([NUM_FEATURES])

        x_image = tf.reshape(input_placeholder, [-1, KX, KY, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    
    # Define second convolution
    """ Lets ignore a second convolution right now
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
    """

    # Define a fully connected layer
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([6 * 6 * NUM_FEATURES, FC_UNITS])
        b_fc1 = bias_variable([FC_UNITS])

        h_pool1_flat = tf.reshape(h_pool1, [-1, 6 * 6 * NUM_FEATURES])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    # TODO Could put Dropout here...

    # Define a fully connected layer
    with tf.name_scope('y_conv'):
        W_fc2 = weight_variable([FC_UNITS, NUM_OUTPUTS])
        b_fc2 = bias_variable([NUM_OUTPUTS])

        y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    

    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    loss = tf.reduce_sum(
        tf.square(label_placeholder - y_conv), name='loss')

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    #correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Collect all summaries together into one operation
    tf.scalar_summary(loss.op.name, loss)
    summary_op = tf.merge_all_summaries()

    sess = tf.Session()

    summary_writer = tf.train.SummaryWriter(TENSORBOARD_DIR,
                                            graph_def=sess.graph_def)

    sess.run(tf.initialize_all_variables())
    for step in range(TOTAL_STEPS):
        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        batch_data = train_dataset[offset : (offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset : (offset + BATCH_SIZE), :]
        feed_dict = { input_placeholder : batch_data, 
                        label_placeholder : batch_labels 
                    }

        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
    
        if step % 1000 == 0:
            print("step:", step, "loss:", loss_value)

        if step % 250 == 0:
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
    """
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    """












    








    """
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

        if step % 250 == 0:
            summary_str, _ = sess.run([summary_op, lamb], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            
            # Save a checkpoint and evaluate the model periodically.
            if step % 10000 == 0:
                #saver.save(sess, SAVE_DIR, global_step=step)
                
                # RUN Network with validation data
                offset = (step * BATCH_SIZE) % (valid_labels.shape[0] - BATCH_SIZE)
                batch_data = valid_dataset[offset : (offset + BATCH_SIZE), :]
                batch_labels = valid_labels[offset : (offset + BATCH_SIZE), :]
                feed_dict = { input_placeholder : batch_data, 
                            label_placeholder : batch_labels 
                            }

                preds, validation_value, = sess.run([logits, loss], feed_dict=feed_dict)
                print("Step: %d: validation: %.5f" % (step, validation_value))
                
                if False: #$step % 100000 == 0:
                    # Save predictions for viewing later
                    pic_name = "g3_" + str(step) 
                    netTools.save_preds(batch_data, preds, pic_name, batch_size=BATCH_SIZE)


    """
