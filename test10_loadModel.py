import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 批次的大小
batch_size = 128
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.matmul(x, W) + b

# 代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer().minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 得到一个布尔型列表，存放结果是否正确
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax 返回一维张量中最大值索引

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把布尔值转换为浮点型求平均数

saver = tf.train.Saver()

model_dir = 'net/8-1'

with tf.Session() as sess:
    sess.run(init)
    acc1 = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    saver.restore(sess, model_dir + '/test_net.ckpt')
    acc2 = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print(" Init Accuracy" + str(acc1))
    print(" Restore Accuracy: " + str(acc2))
