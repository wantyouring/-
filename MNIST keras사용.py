# 정확도 0.92까지.

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

#MNIST 데이터 받아오기. 학습 데이터, 테스트 데이터.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
'''
기존 타입 : 
    x_train : (60000,28,28) uint8
    y_train : (10000,) uint8

변환 후 타입 :
    x_train : (60000,784) float32
    y_train : (10000,10) float32 (원핫인코딩 됨)
'''

# 데이터를 float로 변환 후 스케일링. 이미지 전처리하는 보편적 방법이라고 한다...(?)
# 255로 나누는 이유??? -> 픽셀값을 0~1 값으로 정규화하는 작업.
x_train = x_train.reshape(60000,784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
# np_utils.to_categorical : 원핫인코딩 기능
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#layer 선형 스택
model = Sequential()
# units : 출력 개수, input_dim : 입력 개수, activation : 활성화 함수
model.add(Dense(units=64, input_dim=28*28, activation='relu')) #layer stacking.
model.add(Dense(units=10, activation='softmax'))

# loss함수, optimizer, 평가기준(metrics) -> metrics 지정하면 accuracy도 나옴. 원래는 loss만 나온다.
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
# model.fit()은 학습. (입력 데이터, 라벨, 학습 epoch 횟수, 배치 크기)
# epoch : 학습용 데이터 전체를 한 번 사용했을 시 1epoch
hist = model.fit(x_train, y_train, epochs=5, batch_size=32)

# 5번 epoch 각각의 loss, 정확도 프린트.
print('## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['acc'])

# 모델 평가. loss와 정확도 출력.
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# 모델 예측. 데이터를 넣고 예측값 확인.
xhat = x_test[0:1]
yhat = model.predict(xhat)
print(np.argmax(yhat))
print('## yhat ##')
print(yhat)