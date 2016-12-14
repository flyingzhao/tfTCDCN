from PrepareData import get_next_batch
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


image = tf.placeholder(tf.float32, shape=[None, 40, 40])
landmark = tf.placeholder(tf.float32, shape=[None, 10])
gender = tf.placeholder(tf.float32, shape=[None, 2])
smile = tf.placeholder(tf.float32, shape=[None, 2])
glasses = tf.placeholder(tf.float32, shape=[None, 2])
headpose = tf.placeholder(tf.float32, shape=[None, 5])

# layer 1
W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(image, [-1, 40, 40, 1])

h_conv1 = tf.abs(tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1))
h_pool1 = max_pool_2x2(h_conv1)

# layer2
W_conv2 = weight_variable([3, 3, 16, 48])
b_conv2 = bias_variable([48])
h_conv2 = tf.abs(tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2))
h_pool2 = max_pool_2x2(h_conv2)

# layer3
W_conv3 = weight_variable([3, 3, 48, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.abs(tf.nn.tanh(conv2d(h_pool2, W_conv3) + b_conv3))
h_pool3 = max_pool_2x2(h_conv3)

# layer4
W_conv4 = weight_variable([2, 2, 64, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.abs(tf.nn.tanh(conv2d(h_pool3, W_conv4) + b_conv4))
h_pool4 = h_conv4

#  layer5
W_fc1 = weight_variable([2 * 2 * 64, 100])
b_fc1 = bias_variable([100])

h_pool4_flat = tf.reshape(h_pool4, [-1, 2 * 2 * 64])
h_fc1 = tf.abs(tf.nn.tanh(tf.matmul(h_pool4_flat, W_fc1) + b_fc1))

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer

# landmark
W_fc_landmark = weight_variable([100, 10])
b_fc_landmark = bias_variable([10])
y_landmark = tf.matmul(h_fc1_drop, W_fc_landmark) + b_fc_landmark

# gender
W_fc_gender = weight_variable([100, 2])
b_fc_gender = bias_variable([2])
y_gender = tf.matmul(h_fc1_drop, W_fc_gender) + b_fc_gender
# smile
W_fc_smile = weight_variable([100, 2])
b_fc_smile = bias_variable([2])
y_smile = tf.matmul(h_fc1_drop, W_fc_smile) + b_fc_smile
# glasses
W_fc_glasses = weight_variable([100, 2])
b_fc_glasses = bias_variable([2])
y_glasses = tf.matmul(h_fc1_drop, W_fc_glasses) + b_fc_glasses
# headpose
W_fc_headpose = weight_variable([100, 5])
b_fc_headpose = bias_variable([5])
y_headpose = tf.matmul(h_fc1_drop, W_fc_headpose) + b_fc_headpose


error = 1 / 2 * tf.reduce_sum(tf.square(landmark - y_landmark)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_gender, gender)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_smile, smile)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_glasses, glasses)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_headpose, headpose))+\
        2*tf.nn.l2_loss(W_fc_landmark)+\
        2*tf.nn.l2_loss(W_fc_glasses)+\
        2*tf.nn.l2_loss(W_fc_gender)+\
        2*tf.nn.l2_loss(W_fc_headpose)+\
        2*tf.nn.l2_loss(W_fc_smile)

landmark_error = 1 / 2 * tf.reduce_sum(tf.square(landmark - y_landmark))

# train
train_step = tf.train.AdamOptimizer(1e-3).minimize(error)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for x in range(10000):
        i, j, k, l, m, n = get_next_batch("testing", 50)
        print(x, sess.run(error,
                          feed_dict={image: i, landmark: j, gender: k, smile: l, glasses: m, headpose: n,
                                     keep_prob: 1}))
        sess.run(train_step,
                 feed_dict={image: i, landmark: j, gender: k, smile: l, glasses: m, headpose: n, keep_prob: 0.5})

        o, p, q, r, s, t = get_next_batch("training", 50)

        print("testing", sess.run(error, feed_dict={image: o, landmark: p, gender: q, smile: r, glasses: s, headpose: t,
                                                    keep_prob: 1}))

        print("landmark",sess.run(landmark_error,feed_dict={image: o, landmark: p, gender: q, smile: r, glasses: s, headpose: t,
                                                    keep_prob: 1}))