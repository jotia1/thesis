import cPickle as pickle
import numpy as np
import tensorflow as tf

pickle_file = 'relearn.pickle'

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

################   DEFINING NETWORK #########################

graph = tf.Graph()
batch_size = 40

with graph.as_default():
    
    # Load all the input data as constants on the graph
    #tf_train_dataset = tf.constant(train_dataset)
    #tf_train_labels = tf.constant(train_labels)
    
    tf_train_dataset = tf.placeholder(tf.float32,
            shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32,
            shape=(batch_size, image_size * image_size))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

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
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = hidden_output
    valid_prediction = tf.matmul(tf_valid_dataset, weights) + biases
    test_prediction = tf.matmul(tf_test_dataset, weights) + biases

# Visualisation - TensorBoard
w_hist = tf.histogram_summary("weights", weights)
hid_summ = tf.scalar_summary('Hidden', hidden_output)
merged = tf.merge_all_summaries()

################   ACTUAL TRAINING  ###################

def accuracy(pred, labels):
    return tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(pred, labels), 2))).eval()
    

num_steps = 100

with tf.Session(graph=graph) as session:
    writer = tf.train.SummaryWriter('/home/Student/s4290365/thesis/tf_models/tensorBoard', session.graph_def)
    writer.flush()
    writer.close()
    tf.initialize_all_variables().run()
    print "Initialized"

    for step in xrange(num_steps):
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


