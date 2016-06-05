import numpy as np
import tensorflow as tf
import netTools
import time
import math
#import scipy.io
import h5py

#DATA_FILE = 'arbAng_gt5.mat'
N5_DATA_FILE = '8ang1_30ms_midActv.mat'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
#TENSORBOARD_DIR = 'net4_arbang25m/'
N5_TENSORBOARD_DIR = 'net4_arbang/'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
N5_SAVE_DIR = N5_TENSORBOARD_DIR
N5_MODEL_ID = 'net4_arbang.ckpt'
N5_IMG_DIR = N5_TENSORBOARD_DIR + ''
N5_LOAD_MODEL = True
N5_SAVE_MODEL = False
N5_WRITE_IMAGES = True

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
#IMAGE_SIZE = 128
#TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Hyper params
N5_BATCH_SIZE = 100
N5_TOTAL_STEPS = 1
N5_LEARNING_RATE = 0.1
N5_NUM_HIDDEN_UNITS = 2
N5_ACTIVATION = 'relu'

def runNet(datafile=N5_DATA_FILE, tensorboard_dir=N5_TENSORBOARD_DIR, save_dir=N5_SAVE_DIR, 
            model_id=N5_MODEL_ID, image_dir=N5_IMG_DIR, load_model=N5_LOAD_MODEL, 
            save_model=N5_SAVE_MODEL, write_image=N5_WRITE_IMAGES, batch_size=N5_BATCH_SIZE, 
            total_steps=N5_TOTAL_STEPS, learning_rate=N5_LEARNING_RATE, other_params={}):
    # Load and separate datasets
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
    num_hidden_units = other_params.get('num_hidden_units', N5_NUM_HIDDEN_UNITS) 
    activation = other_params.get('activation', N5_ACTIVATION)

    # NETWORK PARAMS
    num_input_units = kx * ky
    #HIDDEN_UNITS = 16
    num_output_units = kx * ky

    print("INPUT_UNITS:", num_input_units, 
            "HIDDEN_UNITS:", num_hidden_units,
            "OUTPUT_UNITS:", num_output_units)


    # Build graph
    # Using the standard graph
    with tf.Graph().as_default():
        
        # Network inputs and outputs
        input_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            num_input_units))
        label_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            num_output_units))
        
        # Define hidden layer
        with tf.name_scope('hidden_layer'):
            h_weights = tf.Variable(
                tf.truncated_normal([num_input_units, num_hidden_units],
                                    stddev=0.1),
                name='h_weights')

            h_biases = tf.Variable(tf.truncated_normal([num_hidden_units], 
                                    stddev=0.1),
                name='h_biases')
            h_logits = tf.matmul(input_placeholder, h_weights) + h_biases

            # Hidden layer activations?
            if activation == 'relu':
                h_output = tf.nn.relu(h_logits, name='h_output')
            elif activation == 'sigmoid':
                h_output = tf.sigmoid(h_logits, name='h_output')
            elif activation == 'linear':
                h_output = h_logits
            else:
                raise Exception('Activation not known: ' + activation)


        tf.histogram_summary("h_biases", h_biases)
        tf.histogram_summary("h_weights", h_weights)
        tf.histogram_summary("h_logits", h_logits)
        tf.histogram_summary("h_output", h_output)

        # Define output layer
        with tf.name_scope('out_layer'):
            o_weights = tf.Variable(
                tf.truncated_normal([num_hidden_units, num_output_units],
                                    stddev=0.1),
                name='o_weights')

            o_biases = tf.Variable(tf.truncated_normal([num_output_units], 
                                    stddev=0.1),
                name='o_biases')
            o_logits = tf.matmul(h_output, o_weights) + o_biases

        tf.histogram_summary("out_biases", o_biases)
        tf.histogram_summary("out_weights", o_weights)
        tf.histogram_summary("out_logits", o_logits)

        # loss function - Sum squared difference (well mean...)
        loss = tf.reduce_mean(
            tf.square(label_placeholder - o_logits), name='loss')


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

        
        for step in xrange(total_steps):
            # Create feed dictionary
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset : (offset + batch_size), :]
            batch_labels = train_labels[offset : (offset + batch_size), :]
            feed_dict = { input_placeholder : batch_data, 
                        label_placeholder : batch_labels 
                        }

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            if step % 250 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                
# Save a checkpoint and evaluate the model periodically.            
            if step % total_steps - 1 == 0:                                     
                eval_model(step, batch_size, valid_labels,                       
                        valid_dataset, sess, o_logits, loss, 
                        input_placeholder, label_placeholder)  
            """
                if step % total_steps - 1 == 0:
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

    if save_model:
        save_path = saver.save(sess, save_dir + model_id)
        print("Model saved in file: %s" % save_path)


    ## If on my laptop then just write straight to images
    if write_image:
        print("Attempting to write images now...")
        import visTools
        batch_data, batch_labels, preds = eval_model(step, batch_size, valid_labels, 
                        valid_dataset, sess, o_logits, loss, input_placeholder, label_placeholder)
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
    return batch_data, batch_labels, preds                                                                
                  

if __name__ == '__main__':
    runNet()








