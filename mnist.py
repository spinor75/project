import tensorflow as tf
#import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data
 
# loading data
mnist = input_data.read_data_sets("/mnt/c/Users/spinor75/MNIST/", one_hot=True)
tf.set_random_seed(777) 
nb_classes = 10
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,nb_classes])
 
# using softmax classifier
W1 = tf.get_variable("W1",shape=[784,512], initializer=tf.contrib.layers.xavier_initializer())
#W1 = tf.Variable(tf.random_normal([784, 512]))
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X,W1) + b1) 
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2",shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
#W2 = tf.Variable(tf.random_normal([512, 512]))
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2) 
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)


W3 = tf.get_variable("W3",shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
#W3 = tf.Variable(tf.random_normal([512, 512]))
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2,W3) + b3) 
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)


W4 = tf.get_variable("W4",shape=[512,512], initializer=tf.contrib.layers.xavier_initializer())
#W4 = tf.Variable(tf.random_normal([512, 512]))
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3,W4) + b4) 
L4 = tf.nn.dropout(L3, keep_prob=keep_prob)


W5 = tf.get_variable("W5",shape=[512,nb_classes], initializer=tf.contrib.layers.xavier_initializer())
#W5 = tf.Variable(tf.random_normal([512, nb_classes]))
b5 = tf.Variable(tf.random_normal([nb_classes]))
# Hypothesis(using Softmax)
#hypothesis = tf.nn.softmax(tf.matmul(L2, W3) + b3)
hypothesis = tf.matmul(L4, W5) + b5
 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
 

#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
 
# Test model
is_correct = tf.equal(tf.arg_max(hypothesis,1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
 
# parameters
# Training epoch/batch
training_epochs = 15
batch_size = 100
 
with tf.Session() as sess:
    # Init Tensorflow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
 
        for i in range(batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _= sess.run([cost,optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob:0.7})
            avg_cost += c / total_batch
 
        print('Epoch:', '%04d' %(epoch+1), 'cost = ', '{:.9f}'.format(avg_cost))
 
    print("Learning finished")
    print("Accuracy: ", accuracy.eval(session=sess,feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
 
    # Get one and predict using matplotlib
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
 
#    plt.imshow(
#        mnist.test.images[r:r + 1].reshape(28, 28),
#        cmap='Greys',
#        interpolation='nearest')
#    plt.show()


