import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    c1 = tf.constant([1])
g1.as_default()
c2 = tf.constant([2])
# g2 = tf.Graph()
# g2.as_default()
# c2 = tf.constant([2])

with tf.Session(graph=g1) as sess1:
    print(sess1.run(c1))
with tf.Session() as sess:
    print(sess.run(c2))
# with tf.Session(graph=g2) as sess2:
#     print(sess2.run(c2))