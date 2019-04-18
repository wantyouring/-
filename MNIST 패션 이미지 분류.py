# model param 120만.
# 3 epoch학습 -> 정확도 0.911

from keras.datasets import fashion_mnist
from keras.utils import np_utils
# from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# 데이터 로드
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 데이터 형변환
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test, 10)
# 모델 구축(CNN)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()

# 모델 설정
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델 학습
model.fit(x_train,y_train,batch_size=32,epochs=3)

print('\nAccuracy : {}'.format(model.evaluate(x_test,y_test)[1]))