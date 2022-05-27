from tensorflow.keras.layers import Input, Dense, concatenate, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers


class FFNN:
    def __init__(self):
        self.inputs = Input(shape=(10,))
        self.model = None

    def makeModel(self):
        # model.add(Dense(n, input_dim=m, activation='')) 같은 선형적 형태가 아니라, 내가 지정한 형태의 input이 여기로 들어오도록 처리
        hidden1 = Dense(64, activation='relu')(self.inputs)
        hidden2 = Dense(64, activation='relu')(hidden1)
        output = Dense(1, activation='sigmoid')(hidden2)
        self.model = Model(inputs=self.inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def makeModel2(self):
        x = Dense(8, activation='relu')(self.inputs)
        x = Dense(4, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        self.model = Model(self.inputs, x)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    def summary(self):
        self.model.summary()


class VariousModel:
    def __init__(self):
        self.model = None

    def linearRegModel(self):
        X = [i for i in range(1, 10)]
        y = [11, 22, 33, 44, 53, 66, 77, 87, 95]
        inputs = Input(shape=(1,))
        output = Dense(1, activation='linear')(inputs)
        self.model = Model(inputs, output)
        sgd = optimizers.SGD(lr=0.01)

        self.model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
        self.model.fit(X, y, epochs=300)

    def logisticModel(self):
        inputs = Input(shape=(3,))
        output = Dense(1, activation='sigmoid')(inputs)
        self.model = Model(inputs, output)

    def multipleInputModel(self):
        # 두개의 입력층 정의
        inputA = Input(shape=(64,))
        inputB = Input(shape=(128,))

        # 첫번째 입력층으로부터 분기되어 진행되는 인공신경망 정의
        x = Dense(16, activation='relu')(inputA)
        x = Dense(8, activation='relu')(x)
        x = Model(inputs=inputA, outputs=x)

        # 두번쨰 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
        y = Dense(64, activation='relu')(inputB)
        y = Dense(32, activation='relu')(y)
        y = Dense(8, activation='relu')(y)
        y = Model(inputs=inputB, outputs=y)

        # 두개의 인공 신경망의 출력을 연결
        result = concatenate([x.output, y.output])

        z = Dense(2, activation='relu')(result)
        z = Dense(1, activation='linear')(z)

        # inputA와 inputB를 z에 연결하는 것이 아니라 모델 x와 모델 y를 z에 연결시켜야 하므로, input으로 x.input, y.input을 집어넣음.
        self.model = Model(inputs=[x.input, y.input], outputs=z)

    def RNN(self):
        inputs = Input(shape=(50, 1))  # 하나의 특성에 50개의 시점을 가진 입력
        lstm_layer = LSTM(10)(inputs)
        x = Dense(10, activation='relu')(lstm_layer)
        output = Dense(1, activation='sigmoid')
        self.model = Model(inputs=inputs, outputs=output)


if __name__ == "__main__":
    f = FFNN()
    f.makeModel2()
    f.summary()
