import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

k = tf.Variable(0., name='k')
b = tf.Variable(0., name='b')
y = k*x_data + b

loss = tf.reduce_mean(tf.square(y_data-y), name='loss')
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss, name='train')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    # t = sess.graph.get_tensor_by_name('train:0')
    t = sess.graph.get_operation_by_name('train')
    for step in range(201):
        sess.run(t)
        if step % 20 == 0:
            t = sess.graph.get_tensor_by_name('k:0')
            print(step, sess.run([k, b]))
            # print(sess.run(t))
