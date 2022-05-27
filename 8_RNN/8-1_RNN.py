from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
import numpy as np
from scipy.special import softmax


class RNN:
    def __init__(self):
        pass

    def sample(self):
        model = Sequential()
        # input_shape = (timesteps(input_length), input_dim)
        # Wh = 3*3,  Wx = 3*10 , b=3  -> 42
        model.add(SimpleRNN(3, input_shape=(2, 10)))
        model.summary()

    def sample2(self):
        model = Sequential()
        # batch_input_shape = (batch_size, timesteps, input_dim)
        model.add(SimpleRNN(3, batch_input_shape=(8, 2, 10)))
        model.summary()

    def sample3(self):
        model = Sequential()
        # batch_input_shape = (batch_size, timesteps, input_dim)
        # return_sequences : 매 시점마다 은닉상태 출력
        model.add(SimpleRNN(3, batch_input_shape=(8, 2, 10), return_sequences=True))
        model.summary()


class SampleRNN:
    # [사랑을 한다],[연애를 하다],[운동을 열심히 하다],[창밖을 유심히 관찰하여 보다]

    # "사랑을 한다" 와 "연애를 하다" 에 대한 RNN학습 진행
    # 단어벡터의 차원 : 사랑을, 한다, 연애를, 하다, 운동을, 열심히, 하다, 창밖을, 유심히, 관찰하여, 보다 -> 11
    # 시점의 수 : 2
    # batch_size : 2
    # 입력 시퀀스의 길이가 제각각 달라도 같은 batch_size 묶음으로 구성할수 있는건지? -> 패딩을 진행하여 길이 맞추기.

    timesteps = 10  # input_length = timestep (문장의 길이=시점의 수)
    input_dim = 4  # 입력의 차원(단어 벡터의 차원)
    hidden_units = 8  # 은닉상태의 크기(메모리 셀의 용량)

    # 입력에 해당되는 2D텐서
    inputs = np.random.random((timesteps, input_dim))

    # 초기 은닉 상태는 0(벡터)로 초기화
    hidden_state_t = np.zeros((hidden_units,))

    print(hidden_state_t)

    # 이전은닉상태와 현재입력을 받아 "현재 은닉상태"를 생성해내기 위해 기본적으로 은닉셀의 수를 기반으로 각각의 크기가 정해짐.

    Wx = np.random.random((hidden_units, input_dim))  # 입력에 대한 가중치
    Wh = np.random.random((hidden_units, hidden_units))  # 은닉상태에 대한 가중치
    b = np.random.random((hidden_units,))  # 편향(bias)

    print('가중치 Wx의 크기(shape) :', np.shape(Wx))
    print('가중치 Wh의 크기(shape) :', np.shape(Wh))
    print('편향의 크기(shape) :', np.shape(b))

    total_hidden_states = []
    for input_t in inputs:
        # Wx * Xt + Wh * Ht-1 + b(bias)
        output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)

        # 각 시점 t별 메모리 셀의 출력의 크기는 (timestep t, output_dim)
        # 각 시점의 은닉 상태의 값을 계속해서 누적
        total_hidden_states.append(list(output_t))
        hidden_state_t = output_t
        # print(hidden_state_t)

    total_hidden_states = np.stack(total_hidden_states, axis=0)

    # (timesteps, output_dim)
    print('모든 시점의 은닉 상태 :')
    print(total_hidden_states)


class BasicRNN:
    """
    출처: https://junstar92.tistory.com/128 [별준 코딩]

    은닉층 : ht = tahn(Wx*Xt + Wh*Ht-1 +b )
    출력증 : yt = f(WyHt +b)  (단, f는 비선형 활성화 함수중 하나)

    n_x : number of units (입력의 차원 = 단어벡터의 차원)
    m : batches of size
    n_a : number of hidden units (은닉셀의 수)
    n_y : number of output (출력 벡터의 차원)
    """

    def rnn_cell_forward(self, Xt, Ht_1, parameters):
        """
        :param Xt: your input data at timestep "t", numpy array of shape (n_x, m).
        :param Ht_1: Hidden state at timestep "t-1", numpy array of shape (n_a, m)
        :param parameters: -- python dictionary containing:
                    Wx -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                    Wh -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                    Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                    ba -- Bias, numpy array of shape (n_a, 1)
                    by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        :return:
            Ht -- next hidden state, of shape (n_a, m)
            yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
            cache -- tuple of values needed for the backward pass, contains (Ht, Ht_1, Xt, parameters)

        """

        # Retrieve parameters from "parameters"
        Wx = parameters["Wx"]
        Wh = parameters["Wh"]
        Wy = parameters["Wy"]
        bh = parameters["bh"]
        by = parameters["by"]
        # compute next activation state using the formula given above
        Ht = np.tanh(Wx.dot(Xt) + Wh.dot(Ht_1) + bh)
        # compute output of the current cell using the formula given above
        yt_pred = softmax(Wy.dot(Ht) + by)

        # store values you need for backward propagation in cache
        cache = (Ht, Ht_1, Xt, parameters)

        return Ht, yt_pred, cache

    def rnn_forward(self, x, H0, parameters):
        # Initialize "caches" which will contain the list of all caches
        caches = []

        # Retrieve dimensions from shapes of x and parameters["Wya"]
        n_x, m, T_x = x.shape
        n_y, n_a = parameters["Wy"].shape

        # initialize "a" and "y_pred" with zeros (≈2 lines)
        H = np.zeros((n_a, m, T_x))
        y_pred = np.zeros((n_y, m, T_x))

        # Initialize Ht (≈1 line)
        Ht = H0

        # loop over all time-steps of the input 'x'
        for t in range(T_x):
            xt = x[:, :, t]
            Ht, yt_pred, cache = self.rnn_cell_forward(xt, Ht, parameters)
            H[:, :, t] = Ht
            y_pred[:, :, t] = yt_pred
            caches.append(cache)

        caches = (caches, x)

        return H, y_pred, caches

    def test(self):
        np.random.seed(1)
        x_tmp = np.random.randn(3, 10, 4)
        a0_tmp = np.random.randn(5, 10)
        parameters_tmp = {}
        parameters_tmp['Wh'] = np.random.randn(5, 5)
        parameters_tmp['Wx'] = np.random.randn(5, 3)
        parameters_tmp['Wy'] = np.random.randn(2, 5)
        parameters_tmp['bh'] = np.random.randn(5, 1)
        parameters_tmp['by'] = np.random.randn(2, 1)
        a_tmp, y_pred_tmp, caches_tmp = self.rnn_forward(x_tmp, a0_tmp, parameters_tmp)
        print("a[4][1] = \n", a_tmp[4][1])
        print("a.shape = \n", a_tmp.shape)
        print("y_pred[1][3] =\n", y_pred_tmp[1][3])
        print("y_pred.shape = \n", y_pred_tmp.shape)
        print("caches[1][1][3] =\n", caches_tmp[1][1][3])
        print("len(caches) = \n", len(caches_tmp))


if __name__ == "__main__":
    # RNN().sample()
    b = BasicRNN()
    b.test()
