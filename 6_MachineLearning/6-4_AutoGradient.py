import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import time


class TapeGradient:

    def f(self, w):
        y = w ** 2
        z = 2 * y + 5
        return z

    def test(self):
        w = tf.Variable(2.)
        with tf.GradientTape() as tape:
            z = self.f(w)

        gradients = tape.gradient(z, [w])

        print(gradients)


class linearRegression:
    def __init__(self):
        # 학습될 가중치 변수 초기값 설정
        self.w = tf.Variable(3.0)
        self.b = tf.Variable(2.0)

    @tf.function
    def hypothesis(self, x):
        return self.w * x + self.b

    def hypothesisTest(self):
        x_test = [3.5, 5, 5.5, 6]
        print(self.hypothesis(x_test).numpy())

    @tf.function
    def mse_loss(self, y_pred, y):
        # 두 개의 차이값의 제곱을 하여 평균 취함
        return tf.reduce_mean(tf.square(y_pred - y))

    def learn(self):
        x = [n for n in range(1, 10)]  # 공부하는 시간
        # print(x)
        y = [11, 22, 33, 44, 53, 66, 77, 87, 95]  # 공부하는 시간에 맵핑되는 성적

        # 경사하강법 사용, 학습률 0.01
        optimizer = tf.optimizers.SGD(0.01)
        # optimizer = tf.train.GradientDescentOptimizer(0.01)

        for i in range(301):
            with tf.GradientTape() as tape:
                # 현재 파라미터에 기반한 입력 x에 대한 예측값 = y_pred
                y_pred = self.hypothesis(x)

                # 평균 제곱 오차 계산
                cost = self.mse_loss(y_pred, y)

            # 손실 함수에 대해 파라미터의 미분값 계산
            gradients = tape.gradient(cost, [self.w, self.b])

            # 파라미터 업데이트
            optimizer.apply_gradients(zip(gradients, [self.w, self.b]))

            if i % 10 == 0:
                print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, self.w.numpy(),
                                                                                              self.b.numpy(), cost))


class linRegUseKeras:
    def __init__(self):
        self.model = None
        self.x = [n for n in range(1, 10)]  # 공부하는 시간
        self.y = [11, 22, 33, 44, 53, 66, 77, 87, 95]  # 공부하는 시간에 맵핑되는 성적

    def learn(self):
        self.model = Sequential()

        # 출력 y의 차원은 1, 입력 x의 차원(input_dim)은 1
        # 선형 회귀이므로 activation은 'linear'
        self.model.add(Dense(1, input_dim=1, activation='linear'))

        sgd = optimizers.SGD(lr=0.01)

        # 손실 함수(loss function)은 평균제곱오차 'mse'를 사용
        self.model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

        # 주어진 x와 y데이터에 대해 오차를 최소화하는 작업을 300번 시도.
        self.model.fit(self.x, self.y, epochs=300)

    def plot(self):
        print("plot start")
        # 모델에 의해 예측된 값들은 파란색선으로 표현, 원래 데이터값들은 점으로 표현
        plt.plot(self.x, self.model.predict(self.x), 'b', self.x, self.y, 'k.')
        # plt.plot(self.x, self.model.predict(self.x),'b')
        plt.show()
        # time.sleep(10)


if __name__ == "__main__":
    # TapeGradient().test()
    # linearRegression().hypothesisTest()

    # l = linearRegression()
    # l.learn()
    # l.hypothesisTest()

    l = linRegUseKeras()
    l.learn()
    l.plot()
