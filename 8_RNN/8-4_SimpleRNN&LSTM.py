import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional, Input, Dense
from tensorflow.keras.models import Model


class ModelTest:
    def __init__(self):
        # 문장길이가 4, 단어집합 개수(단어벡터)는 5, 샘플이 한개밖에 없으므로 배치크기 1 -> (1,4,5)
        self.train_X = [[[0.1, 4.2, 1.5, 1.1, 2.8], [1.0, 3.1, 2.5, 0.7, 1.1], [0.3, 2.1, 1.5, 2.1, 0.1],
                         [2.2, 1.4, 0.5, 0.9, 1.1]]]
        self.train_X = np.array(self.train_X, dtype=np.float32)
        print(self.train_X.shape)

    def testRnnModel(self):
        rnn = SimpleRNN(3)  # rnn = SImpleRNN(3, return_sequences=False, return_state=False) 와 동일
        hidden_state = rnn(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))

        rnn = SimpleRNN(3, return_sequences=True)
        hidden_state = rnn(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))

        rnn = SimpleRNN(3, return_sequences=True, return_state=True)
        hidden_state, last_state = rnn(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))
        print('last state : {}, shape : {}'.format(last_state, last_state.shape))

        rnn = SimpleRNN(3, return_sequences=False, return_state=True)
        hidden_state, last_state = rnn(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))
        print('last state : {}, shape : {}'.format(last_state, last_state.shape))

    def testLstmModel(self):
        # return_state True이면 last_state와 last_cell_state까지 반환
        lstm = LSTM(3, return_sequences=False, return_state=True)
        hidden_state, last_state, last_cell_state = lstm(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))
        print('last state : {}, shape : {}'.format(last_state, last_state.shape))
        print('last_cell_state : {}, shape : {}'.format(last_cell_state, last_cell_state.shape))
        print('\n')

        lstm = LSTM(3, return_sequences=False, return_state=False)
        hidden_state = lstm(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))
        print('\n')

        lstm = LSTM(3, return_sequences=True, return_state=False)
        hidden_state = lstm(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))
        print('\n')

        lstm = LSTM(3, return_sequences=True, return_state=True)
        hidden_state, last_state, last_cell_state = lstm(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_state, hidden_state.shape))
        print('last state : {}, shape : {}'.format(last_state, last_state.shape))
        print('last_cell_state : {}, shape : {}'.format(last_cell_state, last_cell_state.shape))
        print('\n')

    def testBiLstmModel(self):
        k_init = tf.keras.initializers.Constant(value=0.1)
        b_init = tf.keras.initializers.Constant(value=0)
        r_init = tf.keras.initializers.Constant(value=0.1)

        # return_sequences=False이므로 Many-to-One
        # 문장이 1,2,3,4 순서라면,  forward는 1로 시작해서 4까지학습된 은닉상태가 출력, backward는 4로 시작해서 1까지 학습된 은닉상태가 출력.
        # forward_h와 backward_h가 concatenate 된 값이 최종 은닉상태로 출력.
        bilstm = Bidirectional(
            LSTM(3, return_sequences=False, return_state=True, kernel_initializer=k_init, bias_initializer=b_init,
                 recurrent_initializer=r_init))
        hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_states, hidden_states.shape))
        print('forward_h : {}, shape : {}'.format(forward_h, forward_h.shape))
        print('backward_h : {}, shape : {}'.format(backward_h, backward_h.shape))
        print('\n')

        # 문장이 1,2,3,4 순서 가정
        # 모델의 첫번째 은닉상태는 forward의 1의 은닉상태와 backward의 4~1까지 학습된 은닉상태가 연결
        # 두번쨰 은닉상태 : forward의 1~2 은닉상태 + backward의 4~2까지 학습된 은닉상태 연결
        # 세번째 은닉상태 : forward의 1~3 은닉상태 + backward의 4~3까지 학습된 은닉상태 연결
        # 네번째 은닉상태 : forward의 1~4 은닉상태 + backward의 4의 은닉상태 연결
        bilstm = Bidirectional(
            LSTM(3, return_sequences=True, return_state=True, kernel_initializer=k_init, bias_initializer=b_init,
                 recurrent_initializer=r_init))
        hidden_states, forward_h, forward_c, backward_h, backward_c = bilstm(self.train_X)
        print('hidden state : {}, shape : {}'.format(hidden_states, hidden_states.shape))
        print('forward_h : {}, shape : {}'.format(forward_h, forward_h.shape))
        print('backward_h : {}, shape : {}'.format(backward_h, backward_h.shape))

if __name__ == "__main__":
    # ModelTest().testRnnModel()
    # ModelTest().testLstmModel()
    ModelTest().testBiLstmModel()
