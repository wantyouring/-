import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess: # with블록 밖으로 나가면 세션 자동으로 close
  # run할 그래프에 placeholder가 있으면 모두 feed_dict해줘야 run 가능함!
  result = sess.run([output],feed_dict={input1:[7,5],input2:[3,2]})
  print(result)
