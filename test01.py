
import tensorflow as tf


m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])
m3 = tf.Variable([[3, 3]])
m4 = tf.Variable([[2], [3]])

v1 = tf.Variable(0)

product1 = tf.matmul(m1, m2)
product2 = tf.matmul(m3, m4)

init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for _ in range(0, 5):
        sess.run(tf.assign(v1, tf.add(v1, 1)))
        print(sess.run(v1 ))
