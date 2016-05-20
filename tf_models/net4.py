import numpy as np
import tensorflow as tf
import netTools
import time
import math
import scipy.io

#DATA_FILE = 'arbAng_gt5.mat'
N4_DATA_FILE = '8ang1_30ms_midActv.mat'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
#TENSORBOARD_DIR = 'net4_arbang25m/'
N4_TENSORBOARD_DIR = 'net4_arbang/'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
N4_SAVE_DIR = N4_TENSORBOARD_DIR
N4_MODEL_ID = 'net4_arbang.ckpt'
N4_IMG_DIR = N4_TENSORBOARD_DIR + ''
N4_LOAD_MODEL = True
N4_SAVE_MODEL = False
N4_WRITE_IMAGES = True

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
#IMAGE_SIZE = 128
#TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Hyper params
N4_BATCH_SIZE = 100
N4_TOTAL_STEPS = 1
N4_LEARNING_RATE = 0.1

def runNet(datafile=N4_DATA_FILE, tensorboard_dir=N4_TENSORBOARD_DIR, save_dir=N4_SAVE_DIR, 
            model_id=N4_MODEL_ID, image_dir=N4_IMG_DIR, load_model=N4_LOAD_MODEL, 
            save_model=N4_SAVE_MODEL, write_image=N4_WRITE_IMAGES, batch_size=N4_BATCH_SIZE, 
            total_steps=N4_TOTAL_STEPS, learning_rate=N4_LEARNING_RATE):
    # Load and separate datasets
    data = scipy.io.loadmat(data_file)
    train_dataset = data.get('train_inputs')
    train_labels = data.get('train_labels')
    valid_dataset = data.get('valid_inputs')
    valid_labels = data.get('valid_labels')

    #Get input size
    kx = int(data.get('kx'))
    ky = int(data.get('ky'))

    # NETWORK PARAMS
    num_input_units = kx * ky
    #HIDDEN_UNITS = 16
    num_output_units = kx * ky

    print("INPUT_UNITS:", num_input_units, "OUTPUT_UNITS:", num_output_units)


    # Build graph
    # Using the standard graph
    with tf.Graph().as_default():
        
        # Network inputs and outputs
        input_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            num_input_units))
        label_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            num_output_units))
        
        # Define output layer
        with tf.name_scope('out_layer'):
            weights = tf.Variable(
                tf.truncated_normal([num_input_units, num_output_units],
                                    stddev=0.1),
                name='weights')

            biases = tf.Variable(tf.truncated_normal([num_output_units], 
                                    stddev=0.1),
                name='biases')
            logits = tf.matmul(input_placeholder, weights) + biases

        tf.histogram_summary("out_biases", biases)
        tf.histogram_summary("out_weights", weights)
        tf.histogram_summary("logits", logits)

        # loss function - Sum squared difference (well mean...)
        loss = tf.reduce_mean(
            tf.square(label_placeholder - logits), name='loss')


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
        if not load_model:
            init = tf.initialize_all_variables()
            sess.run(init)
        else:
            saver.restore(sess, SAVE_DIR + model_id)
            print("Restored network from file: %s" % SAVE_DIR + model_id)

        summary_writer = tf.train.SummaryWriter(tensorboard_dir,
                                                graph_def=sess.graph_def)
        
        for step in xrange(total_steps):
            # Create feed dictionary
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset : (offset + batch_size), :]
            batch_labels = train_labels[offset : (offset + batch_size), :]
            feed_dict = { input_placeholder : batch_data, 
                        label_placeholder : batch_labels 
                        }

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            if True: #step % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                
                if True: #step % 10000 == 0:
                    # RUN Network with validation data
                    offset = (step * batch_size) % (valid_labels.shape[0] - batch_size)
                    batch_data = valid_dataset[offset : (offset + batch_size), :]
                    batch_labels = valid_labels[offset : (offset + batch_size), :]
                    feed_dict = { input_placeholder : batch_data, 
                                label_placeholder : batch_labels 
                                }

                    preds, validation_value, = sess.run([logits, loss], feed_dict=feed_dict)
                    print("Step: %d: validation: %.5f" % (step, validation_value))
                    

    if save_model:
        save_path = saver.save(sess, SAVE_DIR + model_id)
        print("Model saved in file: %s" % save_path)


    ## If on my laptop then just write straight to images
    if write_images:
        print("Attempting to write images now...")
        import visTools

        visTools.write_preds(batch_data, batch_labels, preds, image_dir, kx)


if __name__ == '__main__':
    runNet():








