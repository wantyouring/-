import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    #경고메세지 무시

hello = tf.constant("한글 테스트")
sess = tf.Session()
print(sess.run(hello).decode('UTF-8'))  #한글 테스트(utf-8로 디코딩)