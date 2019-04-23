# 선형회귀 기초예제

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #경고메세지 무시

xData = [1,2,3,4,5,6,7]
yData = [1.5,2.5,3.6,4.5,5.5,6.4,7.5]

# W,b는 구할 예측함수의 기울기와 절편. X,Y는 입력할 데이터(placeholder로 feed주기)
W = tf.Variable(tf.random_uniform([1],-50,50))
b = tf.Variable(tf.random_uniform([1],-50,50))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
H = W*X + b #예측값
#(예측값 - Y)^2 이 최소가 되게 optimize하기
cost = tf.reduce_mean(tf.square(H-Y)) #차원 줄이고 평균값 구하기 위해서 사용!
optimizer = tf.train.GradientDescentOptimizer(tf.Variable(0.01))
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5001):
    sess.run(train, feed_dict={X : xData, Y : yData})
    if i%500 == 0:
        print(i, sess.run(W), sess.run(b))
print(sess.run(H, feed_dict={X: 8}))
