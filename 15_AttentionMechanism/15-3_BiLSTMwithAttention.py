from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer
import os


class imdbReview:
    def __init__(self):
        self.vocab_size = 10000
        # 이미 정수인코딩이 된 상태
        (self.X_train, self.y_train), (self.X_test, self.y_test) = imdb.load_data(num_words=self.vocab_size)
        self.max_len = 500

    def preProcessing(self):
        # 매우긴 몇 개의 문장 떄문에 패딩을 너무 많이 하면 학습이 잘 안될수 있음. 평균길이를 체크하여 적당한 패딩길이 설정.
        print('리뷰의 최대 길이 : {}'.format(max(len(l) for l in self.X_train)))
        print('리뷰의 평균 길이 : {}'.format(sum(map(len, self.X_train)) / len(self.X_train)))

        print('리뷰 개수 : {}'.format(len(self.X_train)))

        self.X_train = pad_sequences(self.X_train, maxlen=self.max_len)
        self.X_test = pad_sequences(self.X_test, maxlen=self.max_len)
        print('패딩 후 리뷰 개수 : {}'.format(len(self.X_train)))
        print('패딩 후 리뷰의 최대 길이 : {}'.format(max(len(l) for l in self.X_train)))

    def makeModel(self, option='LEARN'):

        sequence_input = None
        if option == 'TEST':
            # 양방향 LSTM 두층을 사용
            sequence_input = tf.random.uniform(shape=(1,self.max_len), minval=1, maxval=self.vocab_size,
                                               dtype=tf.dtypes.int32)
        elif option == 'LEARN':
            sequence_input = Input(shape=self.max_len, dtype='int32')

        print('type of sequence input : {}'.format(type(sequence_input)))
        print('shape of sequence input : {}'.format(sequence_input.shape))
        # mask_zero=True 로 설정하면 0으로 된 데이터를 패딩으로 보고 False를 반환하고, 그 외에는 True를 반환하는 개념
        embedded_sequences = Embedding(self.vocab_size, 128, input_length=self.max_len, mask_zero=True)(sequence_input)
        print('type of embedded_sequences : {}'.format(type(embedded_sequences)))
        print('shape of embedded_sequences : {}'.format(embedded_sequences.shape))
        # 두번째 층을 위에 쌓아야 하므로 return_sequences=True
        lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(embedded_sequences)

        lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(
            LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)
        print(lstm.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)

        state_h = Concatenate()([forward_h, backward_h])  # 은닉상태
        state_c = Concatenate()([forward_c, backward_c])  # 셀상태

        attention = BahdanauAttention(64)  # 가중치 크기 정의
        print('lstm : type = {}, shape = {}'.format(type(lstm),lstm.shape))
        print('state_h : type = {}, shape = {}'.format(type(state_h),state_h.shape))
        context_vector, attention_weights = attention(lstm, state_h)
        print('type of context_vector : {}'.format(type(context_vector)))
        print('shape of context vector : {}'.format(context_vector.shape))
        dense1 = Dense(20, activation='relu')(context_vector)
        dropout = Dropout(0.5)(dense1)
        output = Dense(1, activation='sigmoid')(dropout)  # category가 2개이므로 0또는 1로 구분
        print('shape of output : {}'.format(output.shape))

        if option == 'LEARN':
            model = Model(inputs=sequence_input, outputs=output)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(self.X_train, self.y_train, epochs=3, batch_size=256,
                                validation_data=(self.X_test, self.y_test), verbose=1)
            print('\n 테스트 정확도: %.4f' % (model.evaluate(self.X_test, self.y_test)[1]))


class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    """
    values : encoder 층의 timestep별 은닉상태
    query : decoder 층의 현재 은닉상태
    """

    # values 내 units크기와 query 내 hidden_size 크기가 서로 다르면? 어떻게 됨?
    # -> 맞추어 주어야 할듯 제대로 계산이 안될듯.
    def call(self, values, query):  # key와 value가 같음
        """
        :param values: 인코더의 모든 시점 은닉상태 (h1~hn)
        :param query: 디코더의 현재시점 은닉상태(St-1)
        :return:
        """
        # query_shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, hidden_size)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights_shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # max_length axis 에 대한 softmax
        print('Type of attention_weights : ', type(attention_weights))
        # context_vector shape after sum == (batch_size, hidden_size)

        print('어텐션 가중치 크기 : ', attention_weights.shape)
        print('인코더 모든시점의 은닉상태 크기 : ', values.shape)
        # 문장내 각각의 단어마다 어텐션가중치가 정해지고, 각 단어에 해당하는 은닉상태 값에 매핑되는 어텐션가중치를 곱함.
        context_vector = attention_weights * values  # 각각 크기를 출력해보고 어떻게 계산되는지 확인해보기.
        print('컨텍스트 벡터 크기 : ', context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


if __name__ == "__main__":
    i = imdbReview()
    i.preProcessing()
    i.makeModel(option='LEARN')
    # i.makeModel(option='TEST')
