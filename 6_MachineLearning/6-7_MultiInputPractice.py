# 다중 입력에 대한 실습
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


class LinearReg:
    def __init__(self):
        self.X = np.array([[70, 85, 11], [71, 89, 18], [50, 80, 20], [99, 20, 10], [50, 10, 10]])  # 3차원 (중간,기말,가산점 점수)
        self.y = np.array([73, 82, 72, 57, 34])  # 최종 성적

        self.model = None

    def makeModel(self):
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=3, activation='linear'))

        sgd = optimizers.SGD(lr=0.0001)
        self.model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

        self.model.fit(self.X, self.y, epochs=2000)

    def predict(self, x):
        return self.model.predict(x)


class LogisticReg:
    def __init__(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])
        self.y = np.array([0, 0, 0, 1, 1, 1])

        self.model = None

    def makeModel(self):
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=2, activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.01)
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

        self.model.fit(self.X, self.y, epochs=3000)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == "__main__":
    # ex = LinearReg()
    # ex.makeModel()
    # print(ex.predict(ex.X))
    #
    # X_test  = np.array([[20,99,10],[40,50,20]])
    # print(ex.predict(X_test))

    logis = LogisticReg()
    logis.makeModel()
    print(logis.predict(logis.X))
