import numpy  # standard python library for numerical calculations
import pandas  # standard python library for reading .csv files
import tensorflow as tf
from datetime import datetime

print(datetime.now())
# Importing Training Data
train_data = pandas.read_csv('./train.csv')

# Importing Testing Data
test_imageset = pandas.read_csv('./test.csv').values

# Extract image values from the train_data
train_images = train_data.iloc[:, 1:].values  # take all values from index-1 column to end (pixel values of the image)
train_images = train_images.astype(numpy.float)  # set values to float so that we can later normalize the pixel values
test_imageset = test_imageset.astype(numpy.float)

# normalizing the train image pixel values to 0 -> 1
train_imageset = numpy.multiply(train_images, 1.0 / 255.0)
print(len(train_imageset))
# normalizing the test image pixel values to 0 -> 1
test_imageset = numpy.multiply(test_imageset, 1.0 / 255.0)

# Get the labels
label = train_data[[0]].values.ravel()  # get the index-0 column containing the labels of the images
labels_unique = numpy.unique(label).shape[0]  # get number of labels = 10 ie., 0,1,2,3,4,5,6,7,8,9

# Create a One-hot-Vector for image labels
index_offset = numpy.arange(label.shape[0]) * labels_unique
one_hot = numpy.zeros((label.shape[0], labels_unique))
one_hot.flat[index_offset + label.ravel()] = 1
train_labelset = one_hot.astype(numpy.uint8)  # setting the type as integer i.e,. 0:[1 0 0 0 0 0 0 0 0 0] etc..

# Creating placeholders for input and output
x = tf.placeholder(tf.float32, shape=[None, 784])  # 784 => image size
y_actual = tf.placeholder(tf.float32, shape=[None, 10])  # 10  => output classes

x_image = tf.reshape(x, [-1, 28, 28, 1])  # resize 1x784 -> 28x28


# Initialize the weights
def variable_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # add noise to prevent symmentry using ReLU functions
    return tf.Variable(initial)


def biasvariable_weight(shape):
    initial = tf.constant(0.1, shape=shape)  # initialize with small positive value to prevent dead neurons
    return tf.Variable(initial)


# Convolution type - 2D
def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # [batch, horizontal stride, vertical stride, -
    # channels] ; padding = zero  so that size of input and output are same


# Max pooling type - 2x2
def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  # [batch, pool size (2x2), channels]
                          strides=[1, 2, 2, 1], padding='SAME')


# First Layer - Convolution
W_layer_1 = variable_weight([5, 5, 1, 32])  # [filter size (5x5), channels, features]
b_layer_1 = biasvariable_weight([32])  # features = 32

O_layer_1 = tf.nn.relu(conv(x_image, W_layer_1) + b_layer_1)
O_pool_1 = max_pooling(O_layer_1)

# Second Layer - Convolution
W_layer_2 = variable_weight([5, 5, 32, 64])  # 5x5 patch size of 32 channels and 64  features
b_layer_2 = biasvariable_weight([64])

O_layer_2 = tf.nn.relu(conv(O_pool_1, W_layer_2) + b_layer_2)
O_pool_2 = max_pooling(O_layer_2)

# Third Layer - Fully Connected
W_fc_1 = variable_weight([7 * 7 * 64, 1024])  # size reduced to 7x7 with 1024 neurons
b_fc_1 = biasvariable_weight([1024])

O_pool_3 = tf.reshape(O_pool_2, [-1, 7 * 7 * 64])  # resize  7x7x64 -> 3,136x1
O_fc_1 = tf.nn.relu(tf.matmul(O_pool_3, W_fc_1) + b_fc_1)

# Drop-out
dropout_prob = tf.placeholder(tf.float32)
O_fc_1_dropout = tf.nn.dropout(O_fc_1, dropout_prob)

# Fouth Layer - Fully Connected
W_fc_2 = variable_weight([1024, 10])
b_fc_2 = biasvariable_weight([10])

y_predicted = tf.matmul(O_fc_1_dropout, W_fc_2) + b_fc_2

# Defining the cost function and setting the learning rate
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual, logits=y_predicted))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)  # optimization using AdamOptimizer
correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_actual, 1))  # returns the index with highest p
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
final_prediction = tf.argmax(y_predicted, 1)

# send train_data in batches
epochs_completed = 0
index_in_epoch = 0
num_examples = train_imageset.shape[0]


def next_batch(batch_size):  # using mini-batches
    global train_imageset
    global train_labelset
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # if no new training data is available, the data is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = numpy.arange(num_examples)
        numpy.random.shuffle(perm)
        train_imageset = train_imageset[perm]
        train_labelset = train_labelset[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_imageset[start:end], train_labelset[start:end]


# Initialize tensorflow parameters
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):  # increase the value and check
    batch_imageset, batch_labelset = next_batch(50)  # each iteration we load 50 examples
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_imageset, y_actual: batch_labelset, dropout_prob: 1.0})
        # print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch_imageset, y_actual: batch_labelset, dropout_prob: 0.5})  # Dropout is active
    # with each connection at final layer is taken with probability of 0.5

lables_predicted = numpy.zeros(test_imageset.shape[0])
for i in range(0, test_imageset.shape[0]):
    lables_predicted[i: (i + 1)] = final_prediction.eval(
        feed_dict={x: test_imageset[i: (i + 1)], dropout_prob: 1.0})  # dropout is inactive as it is used only during
    #  training phase

numpy.savetxt('submission.csv',
              numpy.c_[range(1, len(test_imageset) + 1), lables_predicted],
              delimiter=',',
              header='ImageId,Label',
              comments='',
              fmt='%d')

print(datetime.now())

sess.close()
