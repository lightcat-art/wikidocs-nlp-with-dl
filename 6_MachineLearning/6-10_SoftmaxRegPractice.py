import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class iris:
    def __init__(self):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/06.%20Machine%20Learning/dataset/Iris.csv",
            filename="Iris.csv")

        self.data = pd.read_csv('Iris.csv', encoding='latin1')
        self.data_X = None
        self.data_y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.history = None

    def dataCheck(self):
        print(self.data[:5])
        # 중복을 허용하지 않고, 있는 데이터의 모든 종류를 출력
        print("품종 종류:", self.data['Species'].unique(), sep="\n")
        # sns.set(style="ticks", color_codes=True)
        # g = sns.pairplot(self.data, hue="Species", palette="husl")

        # 각 종과 특성에 대한 연관 관계
        # sns.barplot(self.data['Species'], self.data['SepalWidthCm'], ci=None)
        # plt.show()
        # sns.barplot(self.data['Species'], self.data['SepalLengthCm'], ci=None)
        # plt.show()
        # sns.barplot(self.data['Species'], self.data['PetalLengthCm'], ci=None)
        # plt.show()
        # sns.barplot(self.data['Species'], self.data['PetalWidthCm'], ci=None)
        # plt.show()

        # Species열에서 각 품종이 몇개씩 있는지 확인
        # self.data['Species'].value_counts().plot(kind='bar')
        # plt.show()

    def preprocessing(self):
        # Iris-virginica=0, Iris-setosa=1, Iris-versicolor=2
        self.data['Species'] = self.data['Species'].replace(['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'],
                                                            [0, 1, 2])
        # 라벨 변경 확인
        # self.data['Species'].value_counts().plot(kind='bar')
        # plt.show()

        # print(self.data[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
        # 컬럼정보는 빼고 값만 취한 X 행렬 생성, 특성은 총 4개
        self.data_X = self.data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
        # print(self.data_X)

        # Y 데이터. 예측 대상.
        self.data_y = self.data['Species'].values

        # 훈련데이터 테스트데이터를 8:2로 나눔.
        # random_state = 재현가능(for reproducibility)하도록 난수의 초기값을 설정해주는 것. 설정하면 여러번 수행하더라도 같은 레코드 추출.
        # suffle은 가져온 데이터의 순서를 그대로 잘라서 사용할것인지 한번 섞어서 자를것인지 설정. 디폴트 True
        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(self.data_X, self.data_y,
                                                                                  train_size=0.8, random_state=1)

        # 원-핫 인코딩
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        # print(self.y_train[:5])
        # print(self.y_test[:5])

    def makeModel(self):
        self.model = Sequential()
        self.model.add(Dense(3, input_dim=4, activation='softmax'))

        # 측정항목 함수(merics)는 loss function와 비슷하지만, 측정항목을 평가한 결과는 모델을 학습시키는데 사용되지 않는다는 점에서 다릅니다.
        # 어느 손실 함수나 측정항목(metrics) 함수로 사용할 수 있습니다.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = self.model.fit(self.X_train, self.y_train, epochs=200, batch_size=1,
                                      validation_data=(self.X_test, self.y_test))

    def estimate(self):
        # print(self.history.history) # 훈련데이터 : loss, accuracy, 테스트데이터 : val_loss, val_accuracy
        f, axes = plt.subplots(1, 2)  # 두개 figure 한번에 표현.
        f.set_size_inches((18, 6))  # 전체 figure 크기 조정.
        # plt.subplots_adjust(wspace=0.15, hspace=0.15)

        ax1 = axes[0]
        ax2 = axes[1]
        epochs = range(1, len(self.history.history['accuracy']) + 1)
        ax1.plot(epochs, self.history.history['loss'])
        ax1.plot(epochs, self.history.history['val_loss'])
        ax1.set_title('model_loss')  # plt.title메소드와 같음.
        ax1.set_ylabel('loss')  # plt.ylabel
        ax1.set_xlabel('epochs')  # plt.xlabel
        ax1.legend(['train', 'val'], loc='upper left')  # 범주 표시
        # plt.show()

        ax2.plot(epochs, self.history.history['accuracy'])
        ax2.plot(epochs, self.history.history['val_accuracy'])
        ax2.legend(['train', 'val'], loc='upper left')

        plt.show()

        testEval = self.model.evaluate(self.X_test, self.y_test)  # 손실값과 정확도 get
        print("\n 테스트 손실값 : ", testEval[0])
        print("\n 테스트 정확도 : ", testEval[1])


if __name__ == "__main__":
    i = iris()
    i.preprocessing()
    i.makeModel()
    i.estimate()
