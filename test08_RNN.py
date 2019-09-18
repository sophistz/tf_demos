import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入图片是28*28
n_inputs = 28  # 输入一行的长度 取决于图片一行28像素
max_time = 28  # 一共28行
lstm_size = 100  # 100个隐层单元
n_classes = 10  # 10个分类
batch_size = 50  # 每批次50样本
n_batch = mnist.train.num_examples // batch_size  # 计算一共多少批次

# 这里的none表示第一个维度可以任意，取决于放多少张图片
x = tf.placeholder(tf.float32, [None, 784])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
def RNN(X, Weights, Biases):
    # inputs = [batch_size,max_time,n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本CELL
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], Weights) + Biases)
    return results


# 计算RNN的返回结果
prediction = RNN(x, weights, biases)
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 把结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy=" + str(acc))
