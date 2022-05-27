from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class NeuralNetwork:
    def __init__(self):
        self.model = None

    def makeModel1(self):
        self.model = Sequential()
        # 3개의 입력과 2개의 출력
        self.model.add(Dense(2, input_dim=3, activation='softmax'))

        self.model.summary()  # 6개의 가중치w와 2개의 편향b = 매개변수 총 8개

    def makeModel2(self):
        # 입력이 4개, 은닉층1(8) 은닉층2(8)  출력 3개인 인공신경망 구축
        self.model = Sequential()

        # 4개의 입력과 8개의 출력. 은닉층은 relu 함수로 vanishing gradient 문제 회피
        self.model.add(Dense(8, input_dim=4, activation='relu'))  # params : 32+8

        self.model.add(Dense(8, activation='relu'))  # params : 64+8

        # 3개의 출력. 다중출력이므로 softmax함수 사용.
        self.model.add(Dense(3, activation='softmax'))  # params : 24+3

        # 파라미터 개수 총 139

        self.model.summary()


if __name__ == "__main__":
    NeuralNetwork().makeModel2()
