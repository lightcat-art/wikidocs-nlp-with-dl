# 로지스틱 회귀 실습
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


class useKeras:

    def __init__(self):
        # [-50 -40 -30 -20 -10  -5   0   5  10  20  30  40  50]
        self.x = np.concatenate(
            ([i for i in np.arange(-50, 0, 10)], [n for n in np.arange(-5, 10, 5)], [k for k in np.arange(10, 60, 10)]))
        # print(self.x)
        # [0 0 0 0 0 0 0 0 1 1 1 1 1]
        self.y = np.concatenate((np.zeros(8, dtype=int), np.ones(5, dtype=int)))
        # print(self.y)

        self.model = None

    def makeModel(self):
        # m = self.model
        # 간단한 참조 변수를 만들어서 편하게 하고싶은데 다시 반영하지 않으면 참조 안되네..
        # 아.. list 안에 있는 객체를 가져온것이 아니고 변수 그대로 반영한거라서 반영안되는건 자바랑 똑같은듯
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=1, activation='sigmoid'))

        sgd = optimizers.SGD(lr=0.05)  # 책에선 0.01로 하였을때 잘 학습되었는데 나는 0.05로 해야 학습잘되네

        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])

        self.model.fit(self.x, self.y, epochs=400)

        # self.model = m

    def plotModelPredict(self):
        plt.plot(self.x, self.model.predict(self.x), 'b', self.x, self.y, 'k.')
        plt.show()

    def predict(self, x):
        print(self.model.predict(x))


if __name__ == "__main__":
    uk = useKeras()
    uk.makeModel()
    # uk.plotModelPredict()
    uk.predict([5])  # 학습때 1차원이라고 지정했는데, 예측때 array-like 변수를 넣어야지만 결과가 나오는듯?
    uk.predict([10])
    uk.predict([1, 2, 3, 4, 4.5])
    uk.predict([11, 21, 31, 41, 500])
    # uk.predict(5) #에러
