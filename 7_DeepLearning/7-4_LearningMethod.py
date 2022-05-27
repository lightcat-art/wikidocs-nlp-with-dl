from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


class LossFunction:
    def __init__(self):
        self.model = None
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=1, activation='sigmoid'))

    def mse(self):
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        # 또는
        # compile의 loss는 tf.keras.losses.Loss를 호출
        self.model.compile(optimizer='adma', loss=tf.keras.losses.MeanSquareError(), metrics=['mse'])

    def binaryCrossEntropy(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        # OR
        self.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossEntropy(), metrics=['acc'])

    def categoricalCrossEntropy(self):
        # 각 샘플이 여러개의 레이블(class)에 속할 수 있는 경우에 원핫 인코딩을 하여 사용해야하므로, 이것을 사용.
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        # OR
        self.model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])

    def sparseCategorical(self):
        # 레이블에 대해서 원-핫 인코딩 과정을 생략하고, 정수값을 가진 레이블에 대해서 다중 클래스 분류를 수행.
        # 각 샘플이 오직 하나의 class에 속할 때 사용.
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])
        # OR
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
