import numpy as np
import tensorflow as tf
#import netTools
import time
import math
#import scipy.io
import h5py

N6_DATA_FILE = '8ang1_30ms.mat'
#TENSORBOARD_DIR = '/home/Student/s4290365/thesis/tf_models/tensorBoard'
N6_TENSORBOARD_DIR = 'tensorBoard'
#SAVE_DIR = '/home/Student/s4290365/thesis/tf_models/saveDir'
N6_SAVE_DIR = N6_TENSORBOARD_DIR                                                
N6_MODEL_ID = 'net3_arbang.ckpt'                                                
N6_IMG_DIR = N6_TENSORBOARD_DIR + ''                                            
N6_LOAD_MODEL = True                                                            
N6_SAVE_MODEL = False                                                           
N6_WRITE_IMAGES = True  

# CONSTANTS
TRAIN_INDEX = 0
VALID_INDEX = 1
TEST_INDEX = 2
#IMAGE_SIZE = 128
#TOTAL_PIXELS = IMAGE_SIZE * IMAGE_SIZE

N6_BATCH_SIZE = 100
N6_TOTAL_STEPS = 500001
#HIDDEN_UNITS = 16
N6_LEARNING_RATE = 0.1

# CONVOLUTION params
# TODO dynamically congifure this? inc. in mat file?
N6_KX = 11
N6_KY = 11
N6_KZ = 1  # Use depth 1 (gray scale (decayed) images)
N6_TOTAL_VOXELS = N6_KX * N6_KY * N6_KZ
N6_NUM_FEATURES = 9
N6_CONV_SIZE = 6
N6_FC_UNITS = 64  # TODO why?
N6_NUM_INPUTS = N6_TOTAL_VOXELS
N6_NUM_OUTPUTS = N6_TOTAL_VOXELS
N6_ACTIVATION = 'linear'
N6_KERNEL_FILE = ''

def runNet(datafile=N6_DATA_FILE, tensorboard_dir=N6_TENSORBOARD_DIR, 
    save_dir=N6_SAVE_DIR, model_id=N6_MODEL_ID, image_dir=N6_IMG_DIR, 
    load_model=N6_LOAD_MODEL, save_model=N6_SAVE_MODEL, 
    write_image=N6_WRITE_IMAGES, batch_size=N6_BATCH_SIZE,
    total_steps=N6_TOTAL_STEPS, learning_rate=N6_LEARNING_RATE, other_params={}):        
    
    # Load and separate datasets                                                
    data = h5py.File(datafile)                                                  
    train_dataset = np.transpose(data.get('train_inputs'))                      
    train_labels = np.transpose(data.get('train_labels'))                       
    valid_dataset = np.transpose(data.get('valid_inputs'))                      
    valid_labels = np.transpose(data.get('valid_labels'))                       
    test_dataset = np.transpose(data.get('test_inputs'))                        
    test_labels = np.transpose(data.get('test_labels'))

    kx = int(data.get('kx')[0]) 
    ky = int(data.get('ky')[0]) 
    num_inputs = kx * ky 
    num_outputs = kx * ky 
    
    num_features = other_params.get('num_features', N6_NUM_FEATURES)
    conv_size = other_params.get('conv_size', N6_CONV_SIZE)
    fc_units = other_params.get('fc_units', N6_FC_UNITS)
    activation = other_params.get('activation', N6_ACTIVATION)
    kernel_file = other_params.get('kernel_file', N6_KERNEL_FILE)

    # Load kernels
    kernel_file = other_params.get('kernel_file', N6_KERNEL_FILE)
    kernel_data = h5py.File(kernel_file)                                                  
    loaded_kernels = np.transpose(kernel_data.get('kernels'))
    loaded_kernel_size = np.transpose(kernel_data.get('kx'))

    # Ensure there isn't any monkey business
    assert loaded_kernel_size == conv_size
    

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
                                shape=[batch_size, num_inputs])
        label_placeholder = tf.placeholder(tf.float32, 
                                shape=[batch_size, num_inputs])

        # Define first convolution
        with tf.name_scope('conv'):
            #W_conv = weight_variable([conv_size, conv_size, 1, num_features])
            W_conv = tf.Constant(loaded_kernels)
            b_conv = bias_variable([num_features])

            x_image = tf.reshape(input_placeholder, [-1, kx, ky, 1])

            h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
            h_pool = max_pool_2x2(h_conv)

        #tf.histogram_summary("conv_biases", b_conv)                              
        #tf.histogram_summary("conv_weights", W_conv)                            
    
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
            dim = (kx // 2) * (ky //2) * num_features  # shrink after pooling
            W_fc = weight_variable([dim, fc_units])
            b_fc = bias_variable([fc_units])

            h_pool1_flat = tf.reshape(h_pool, [-1, dim])
            h_fc = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc) + b_fc)

        tf.histogram_summary("fc_biases", b_fc)                              
        tf.histogram_summary("fc_weights", W_fc)                            
        tf.histogram_summary("fc_layer", h_fc) 

        # TODO Could put Dropout here...

        # Define output layer
        with tf.name_scope('out_layer'):
            W_out = weight_variable([fc_units, num_outputs])
            b_out = bias_variable([num_outputs])

        if activation == 'relu':
            out_layer = tf.nn.relu(tf.add(tf.matmul(h_fc, W_out), b_out), name='out_layer')
        elif activation == 'sigmoid':
            out_layer = tf.sigmoid(tf.add(tf.matmul(h_fc, W_out), b_out), name='out_layer')
        elif activation == 'linear':
            out_layer = tf.add(tf.matmul(h_fc, W_out), b_out, name='out_layer')
        else:
            raise Exception('Network activation not know: ' + activation)
           
            
        

        
        tf.histogram_summary("out_biases", b_fc)                              
        tf.histogram_summary("out_weights", W_fc)                            
        tf.histogram_summary("out_layer", out_layer) 

        #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        loss = tf.reduce_sum(
            tf.square(label_placeholder - out_layer), name='loss')

        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        #correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Collect all summaries together into one operation
        tf.scalar_summary(loss.op.name, loss)
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter(tensorboard_dir,
                                            graph_def=sess.graph_def)

        sess.run(tf.initialize_all_variables())
        for step in range(total_steps):
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

            if step % total_steps - 1 == 0:                                     
                eval_model(step, batch_size, valid_labels, valid_dataset, 
                            sess, out_layer, loss, input_placeholder, 
                            label_placeholder)

        """
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print("test accuracy %g"%accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        """
        if save_model:  
            save_path = saver.save(sess, save_dir + model_id)  
            print("Model saved in file: %s" % save_path) 
                                                                                
        ## If on my laptop then just write straight to images 
        if write_image:  
            print("Attempting to write images now...")  
            import visTools                               
            batch_data, batch_labels, preds = eval_model(step, batch_size, 
                            valid_labels, valid_dataset, sess, out_layer, loss, 
                            input_placeholder, label_placeholder)
            visTools.write_preds(batch_data, batch_labels, preds, 
                                image_dir, kx) 
                                                                                
                                                                                
def eval_model(step, batch_size, valid_labels, valid_dataset, sess, y_conv, 
                loss, input_placeholder, label_placeholder):
    # RUN Network with validation data                                          
    offset = (step * batch_size) % (valid_labels.shape[0] - batch_size)         
    batch_data = valid_dataset[offset : (offset + batch_size), :]               
    batch_labels = valid_labels[offset : (offset + batch_size), :]              
    feed_dict = { input_placeholder : batch_data,                               
              label_placeholder : batch_labels                                  
              }                                                                 
                                                                                
    preds, validation_value, = sess.run([y_conv, loss], feed_dict=feed_dict)    
    print("Step: %d: validation: %.5f" % (step, validation_value))              
    return batch_data, batch_labels, preds 











    








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
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset : (offset + batch_size), :]
        batch_labels = train_labels[offset : (offset + batch_size), :]
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
                offset = (step * batch_size) % (valid_labels.shape[0] - batch_size)
                batch_data = valid_dataset[offset : (offset + batch_size), :]
                batch_labels = valid_labels[offset : (offset + batch_size), :]
                feed_dict = { input_placeholder : batch_data, 
                            label_placeholder : batch_labels 
                            }

                preds, validation_value, = sess.run([logits, loss], feed_dict=feed_dict)
                print("Step: %d: validation: %.5f" % (step, validation_value))
                
                if False: #$step % 100000 == 0:
                    # Save predictions for viewing later
                    pic_name = "g3_" + str(step) 
                    netTools.save_preds(batch_data, preds, pic_name, batch_size=batch_size)


    """
