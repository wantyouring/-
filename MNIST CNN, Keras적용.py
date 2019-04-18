'''
https://neurowhai.tistory.com/133 참고
1epoch로 수행한 결과 => Accuracy : 0.9836
(?)CNN 적용한 모델과 적용하지 않은 모델의 성능 차이를 어떤 기준으로 비교해야하는가?
1epoch단위면 수행시간이 너무 많이 차이남.
'''

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# mnist 데이터 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# CNN사용하기 위해서 데이터 형변환. 흑백사진이면 마지막 차원 1, RGB면 3이라 함.
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

# y 원핫인코딩
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test, 10)

# CNN 모델 구축하기
model = Sequential()
# 32개 필터, 3*3크기의 필터, input_shape(행,열,색상), relu 활성화함수
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
# 맥스풀링으로 2는 전체 크기 절반으로 줄임. 맥스풀링 : 영역 안에서 최댓값만 남기고 버리기
model.add(MaxPooling2D(pool_size=2))
# 랜덤하게 일부 노드 끄기. 오버피팅 방지
model.add(Dropout(0.25))
# 2차원 배열을 1차원 배열로
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

# 모델 설정
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 모델 학습
#history = model.fit(x_train,y_train,batch_size=200,epochs=3,verbose=0,validation_data=(x_test,y_test))
history = model.fit(x_train,y_train,batch_size=32,verbose=1)

print('\nAccuracy : {}'.format(model.evaluate(x_test,y_test)[1]))