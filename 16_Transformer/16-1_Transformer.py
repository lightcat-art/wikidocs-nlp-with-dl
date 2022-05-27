import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angle(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        # insert another axis using tf.newaxis
        angle_rads = self.get_angle(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                    i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                    d_model=d_model)

        # angle_rads 크기 (position,d_model)
        # 배열의 짝수 인덱스(2i)는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines

        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        # inputs shape는 보통 (batch_size, input_length(position), embeddingDim(d_model)) 일텐데
        # 왜 input length에 해당하는 부분에 axis를 새로 만드는걸까? inputs shape에 대해 다시 확인 필요.
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model



class Encoder:

    def scaled_dot_product_attention(self, query, key, value, mask):
        # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        # padding_mask : (batch_size, 1, 1, key의 문장 길이)

        # Q와 K의 곱. 어텐션 스코어 행렬
        # matmul_qk 크기 : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
        matmul_qk = tf.matmul(query, key, transpose_b=True)

        # 스케일링
        # dk의 루트값으로 나눠줌
        # key의 마지막 shape는 d_model/num_heads = dk 임
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        # 마스킹. 에턴션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
        # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 됨.
        if mask is not None:
            logits += (mask * -1e9)

        # 소프트 맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행
        # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
        attention_weights = tf.nn.softmax(logits, axis=-1)

        # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
        # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        # output 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # 단 key의 문장길이와 value의 문장길이가 같음을 전제 (일반적인 어텐션 매커니즘은 key와 value의 data가 같기 때문에 큰 문제 x)
        # key와 value의 임베딩벡터크기(d_model)나, dk(d_model/num_heads) 크기는 같지 않더라도 문장길이 레벨은 크기가 같아야함!!
        output = tf.matmul(attention_weights, value)

        return output, attention_weights

    def encoder(self, vocab_size, d_model, name="inputs"):
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        print(inputs.shape)
        embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
        print(embeddings.shape)


class TestPosEncoding:
    def process(self):
        sample_pos_encoding = PositionalEncoding(500, 128)
        plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, 128))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()


class TestScaledDocProductAttention:
    def process(self):
        # 임의의 Query, Key, Value인 Q, K, V 행렬 생성
        np.set_printoptions(suppress=True)
        temp_k = tf.constant([[10, 0, 0],
                              [0, 10, 0],
                              [0, 0, 10],
                              [0, 0, 10]], dtype=tf.float32)  # (4, 3)

        temp_v = tf.constant([[1, 0],
                              [10, 0],
                              [100, 5],
                              [1000, 6]], dtype=tf.float32)  # (4, 2)

        temp_q = tf.constant([[10, 10, 0], [0, 10, 0]], dtype=tf.float32)  # (1, 3)

        # 함수 실행
        temp_out, temp_attn = Encoder().scaled_dot_product_attention(temp_q, temp_k, temp_v, None)

        # 1. q와 k의 유사도를 구하는데, k의 행을 스캔하면서 단어별 유사도(어텐션 분포) 계산
        # 2. 구한 어텐션스코어를 가지고 v에 대입해 최종 어텐션 스코어 계산
        # 3. 최종적으로 어텐션 스코어의 shape는 (query의 문장 길이, value의 임베딩벡터(d_model) OR d_model/num_heads)

        # temp_q가 [[10, 10, 0], [0, 10, 0]] 일때
        # q의 첫번째 단어는 k의 첫번째와 두번째 단어와 어텐션 분포 유효값을 가지고 그 결과 v의 첫번쨰 두번쨰 상태를 참조하여 계산
        # q의 두번째 단어는 k의 두번째 단어와만 어텐션 분포 유효값을 가지고 있으며, 그 결과 v의 두번째 상태만을 가지고옴.

        print(temp_attn)  # 어텐션 분포(어텐션 가중치의 나열)
        print(temp_out)  # 어텐션 값


if __name__ == "__main__":
    # Encoder().encoder(vocab_size=10000, d_model=256)
    TestScaledDocProductAttention().process()
