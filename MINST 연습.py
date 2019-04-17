# MNIST 연습
# matmul(x,W)자리 모두 바꾸면?

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./samples/MNIST_data/",one_hot=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #경고메세지 무시
#feeding할 x선언(뒤에 y_선언함)
x = tf.placeholder(tf.float32, [None, 784])

#모델 파라미터
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#softmax 모델 구현, y_선언
y = tf.nn.softmax(tf.matmul(x,W) + b) #x,W자리 바껴있음.
y_ = tf.placeholder(tf.float32, [None, 10])

# 비용함수 선언. cross entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 비용함수를 경사하강법으로 최소화하는 step 선언
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 변수 초기화, 세션 선언해 모델 run
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 1000번 학습
for i in range(1000) :
    batch_xs, batch_ys = mnist.train.next_batch(100)    #100개 데이터 세트로 가져옴
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})

# 정확성 체크
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #[T,F,T,T] 이런식으로 bool list로 나옴
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))