import numpy as np
import matplotlib.pyplot as plt


class sigmoid:

    def sigmoid(self, x):
        # H(x) = 1/(1+exp(-wx+b)) 에서 w=1 b=0임을 가정
        return 1 / (1 + np.exp(-x))

    def test(self):
        # -5부터 5까지 0.1 간격으로 나열된 실수 리스트 반환
        x = np.arange(-5.0, 5.0, 0.1)
        print(x)
        y = self.sigmoid(x)

    def test2(self):
        x = np.arange(-5.0, 5.0, 0.1)
        y1 = self.sigmoid(0.5 * x)
        y2 = self.sigmoid(x)
        y3 = self.sigmoid(2 * x)
        y4 = self.sigmoid(2 * x - 2)  # x축 bias
        self.plot(x, y1, 'r', lineStyle='--')
        self.plot(x, y2, 'g')
        self.plot(x, y3, 'b', lineStyle='--')
        self.plot(x, y4, 'r', lineStyle='--')
        self.plotShow()

    def plot(self, x, y, option, lineStyle=None):
        if lineStyle is None:
            plt.plot(x, y, option)
        else:
            plt.plot(x, y, option, linestyle=lineStyle)

    def plotShow(self):
        plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
        plt.title("Sigmoid Function")
        plt.show()


if __name__ == "__main__":
    sigmoid().test2()
