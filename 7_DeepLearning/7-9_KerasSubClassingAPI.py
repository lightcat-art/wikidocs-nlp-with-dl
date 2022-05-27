import tensorflow as tf


# Model 자체를 내가 원하는대로 구성할수 있을듯? 하지만 Model의 특성을 좀 잘알아야 할듯.
class LinearRegression(tf.keras.Model):
    # 모델의 구조와 동적을 정의하는 생성자를 정의
    def __init__(self):
        # tf.keras.Model 클래스의 속성들을 가지고 초기화
        super(LinearRegression, self).__init__()
        self.linear_layer = tf.keras.layers.Dense(1, input_dim=1, activation='linear')

    # 모델이 데이터를 입력받아 예측값을 리턴하는 포워드 연산 진행
    # 어떤 상황에 쓰는거지? fit 하고 predict만 쓰는것같은데..
    def call(self,x):
        y_pred = self.linear_layer(x)

        return y_pred

class makeModel:
    def __init__(self, model):
        self.model = model

    def linearReg(self):
        X = [i for i in range(1, 10)]
        y = [11, 22, 33, 44, 53, 66, 77, 87, 95]
        sgd = tf.keras.optimizers.SGD(lr=0.01)
        model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
        model.fit(X,y, epochs=10)
        return model



if __name__=="__main__":
    model = LinearRegression()
    modelManager = makeModel(model)
    model = modelManager.linearReg()
    print(model.predict([i for i in range(1,10)]))


