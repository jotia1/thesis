import numpy as np
import tensorflow as tf
import netTools
import time
import math
import scipy.io
import h5py

#ND_DATA_FILE = 'arbAng_gt5.mat'
ND_DATA_FILE = 'testDemo.mat'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
#TENSORBOARD_DIR = 'net4_arbang25m/'
ND_TENSORBOARD_DIR = 'testDemo/'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
ND_SAVE_DIR = 'attn_batch_results/6/6/165/'
ND_MODEL_ID = 'exp8AD_attn_6_6_165k_exp'
ND_IMG_DIR = ND_TENSORBOARD_DIR + ''
ND_LOAD_MODEL = True
ND_SAVE_MODEL = False
ND_WRITE_IMAGES = True

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
#IMAGE_SIZE = 128
#TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Hyper params
ND_BATCH_SIZE = 100
ND_TOTAL_STEPS = 1
ND_LEARNING_RATE = 0.1

def runNet(datafile=ND_DATA_FILE, tensorboard_dir=ND_TENSORBOARD_DIR, save_dir=ND_SAVE_DIR, 
            model_id=ND_MODEL_ID, image_dir=ND_IMG_DIR, load_model=ND_LOAD_MODEL, 
            save_model=ND_SAVE_MODEL, write_image=ND_WRITE_IMAGES, batch_size=ND_BATCH_SIZE, 
            total_steps=ND_TOTAL_STEPS, learning_rate=ND_LEARNING_RATE):
    # Load and separate datasets
    data = h5py.File(datafile)  
    inputs = np.transpose(data.get('inputs'))
    labels = np.transpose(data.get('labels'))
    print(inputs.shape)
    num_samples = inputs.shape[0]


    #Get input size
    kx = int(data.get('kx')[0])
    ky = int(data.get('ky')[0])

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
            saver.restore(sess, save_dir + model_id)
            print("Restored network from file: %s" % save_dir + model_id)

        summary_writer = tf.train.SummaryWriter(tensorboard_dir,
                                                graph_def=sess.graph_def)
        
        result = np.zeros(inputs.shape)
        
        batch_num = 0
        while batch_num < int(num_samples / batch_size):
            batch_data = inputs[batch_num * batch_size : (batch_num + 1) * batch_size]
            feed_dict = { input_placeholder : batch_data
                        }

            preds = sess.run(logits, feed_dict=feed_dict) 
            result[batch_num * batch_size : (batch_num + 1) * batch_size] = preds
            batch_num += 1

        # Then only do the last few
        offset = batch_num * batch_size
        offset_end = offset + (num_samples % batch_size)
        print(offset, offset_end, inputs.shape, batch_data.shape)
        print(inputs[offset : offset_end].shape)
        print(batch_data[offset : offset_end].shape)
        batch_data[0 : num_samples % batch_size] = inputs[offset : offset_end]
        feed_dict = { input_placeholder : batch_data
                    }

        preds = sess.run(logits, feed_dict=feed_dict)
        result[offset : offset_end] = preds[0 : num_samples % batch_size]

        scipy.io.savemat('preds.mat', dict(result=result))
        #np.savez(datafile.strip('.mat'), preds=preds)

        """
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
        """         



if __name__ == '__main__':
    runNet()








