import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Embedding, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


def scaled_dot_product_attention(query, key, value, mask):
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
    print("before masking : {}".format(logits))
    if mask is not None:
        print("mask info : {}".format(mask))
        logits += (mask * -1e9)
    print("after masking : {}".format(logits))

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


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    """ scaled dot product attention에서 (batch_size, num_heads, query의 문장 길이, key의 문장 길이) 에 적용하기 위한것인데
    batch_size도 상관없으므로 (1,1,1,key의 문장길이)로 적용하는것도 괜찮을거 같은데?
    
    -> batch_size별 패딩마스크가 다를수 있다~
    
    input이 뭐길래 (batch_size와 key의 문장길이) 로 들어오는거지 -> key의 토큰화 및 패딩된 배열. 
    key 데이터의 패딩토큰위치에 마스킹을 하는것임."""
    return mask[:, tf.newaxis, tf.newaxis, :]


# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
def create_look_ahead_mask(x):
    # input x 크기 : (batch_size, input의 문장 길이)
    seq_len = tf.shape(x)[1]
    # 크기 (query 문장길이, key문장길이) 에 대한 하삼각행렬 생성
    # -> 사실상 디코더의 첫번째 서브층에만 look_ahead_mask가 적용되므로, query이든 key이든 문장길이는 똑같음.
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    # print('look_ahead_mask shape : {}'.format(look_ahead_mask.shape))
    padding_mask = create_padding_mask(x)  # 패딩마스크도 포함
    # print('padding_mask shape : {}'.format(padding_mask))
    # padding_mask와 look_ahead_mask의 텐서차원이 다르지만,
    # look_ahead_mask = (None, None) = (query 문장길이, key 문장 길이)
    # padding_mask = (None, 1, 1, None) = (batch_size, num_heads, query 문장 길이, key 문장 길이)
    # padding_mask의 뒷부분 2개 차원과 look_ahead_mask를 비교. query 문장길이는 look_ahead_mask가 더 크므로 이것을 따라감.
    # -> 최종 비교결과텐서 shape = (None, 1, None, None) = (batch_size, num_heads, query 문장 길이, key 문장 길이)
    return tf.maximum(look_ahead_mask, padding_mask)  # 두 행렬 모두 포함하도록 maximum함수 이용


def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    # (batch_size, input_length, d_model)
    inputs = Input(shape=(None, d_model), name="inputs")
    # 인코더는 패딩 마스크 사용
    # 패딩마스크는 단어토큰별로 처리되므로 embedding_dim 차원까지는 필요없음 input_length 차원까지 있으면 됨)
    # (batch_size, num_heads, query 문장길이, key 문장길이)
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")

    # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs, 'key': inputs, 'value': inputs,  # Q=K=V
        'mask': padding_mask
    })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention = Dropout(rate=dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs + attention)

    # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
    outputs = Dense(units=dff, activation='relu')(attention)
    outputs = Dense(units=d_model)(outputs)

    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(outputs + attention)

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    # 크기 (batch_size, input_length) : Embedding Layer에 들어갈때는 원핫인코딩할필요 없음.
    inputs = Input(shape=(None,), name="inputs")
    # 크기 (batch_size, num_heads, query 문장길이, key 문장길이)
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")
    # 위 두개의 Model Input은 실제 값이 아닌데도 내부에 정의된 tensorflow Model의 Input으로 들어갈수 있음.
    # 가장 바깥쪽에 있는 곳에서 실제 데이터가 들어가기만 하면 Model Input은 내부 모델의 Input으로 들어갈수 있는것으로 보임.
    print('encoder : inputs shape = {}'.format(inputs.shape))
    print('encoder : padding_mask shape = {}'.format(padding_mask.shape))

    # 포지셔널 인코딩 + 드롭아웃
    embeddings = Embedding(vocab_size, d_model)(inputs)
    # 저자의 의견 : 트랜스포머 공식 구현체를 보면 임베딩 벡터를 임베딩 벡터 사이즈의 루트만큼 나눠주고 있습니다. 그래디언트 배니싱 문제를 완화하는 테크닉으로 보여집니다.
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    print('encoder : before PE embeddings shape = {}'.format(embeddings.shape))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = Dropout(rate=dropout)(embeddings)

    # 인코더를 num_layers 개 쌓기
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                name="encoder_layer_{}".format(i))([outputs, padding_mask])

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    # 크기 (batch_size, input_length, d_model)
    inputs = Input(shape=(None, d_model), name="inputs")
    # 크기 (batch_size, input_length, d_model)
    enc_outputs = Input(shape=(None, d_model), name="encoder_outputs")
    # print('decoder_layer : inputs shape = {}, enc_outputs shape = {}'.format(inputs.shape, enc_outputs.shape))
    # 룩어헤드 마스크(첫번째 서브층) : 크기 (batch_size, num_heads, query 문장길이, key 문장길이)
    look_ahead_mask = Input(shape=(1, None, None), name="look_ahead_mask")
    # print('decoder_layer : look_ahead_mask shape = {}'.format(look_ahead_mask.shape))

    # 패딩 마스크(두번째 서브층) : 크기 (batch_size, num_heads, query 문장길이, key 문장길이)
    # query 문장에 상관없이 key 문장 기준으로 패딩마스크를 씌우는 것이므로 query 문장길이 차원은 1이어도 됨.
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")
    # print('decoder_layer : padding_mask shape = {}'.format(padding_mask.shape))

    # 멀티-헤드 어텐션 (첫번째 서브층/마스크드 셀프 어텐션)
    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1")(
        inputs={'query': inputs, 'key': inputs, 'value': inputs,  # Q=K=V
                'mask': look_ahead_mask})

    # 잔차 연결과 층 정규화
    attention1 = LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    # 멀티-헤드 어텐션(두번째 서브층/디코더-인코더 어텐션)
    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2") \
        (inputs={'query': inputs, 'key': enc_outputs, 'value': enc_outputs,  # Q!=K=V
                 'mask': padding_mask})

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention2 = Dropout(rate=dropout)(attention2)
    attention2 = LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
    outputs = Dense(units=dff, activation='relu')(attention2)
    outputs = Dense(units=d_model)(outputs)

    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)


def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name='decoder'):
    # 크기 (batch_size, input_length) : Embedding Layer에 들어갈때는 원핫인코딩할필요 없음.
    inputs = Input(shape=(None,), name='inputs')
    # 크기 (batch_size, input_length, d_model)
    enc_outputs = Input(shape=(None, d_model), name='encoder_outputs')
    print('decoder : inputs shape = {}, enc_outputs shape = {}'.format(inputs.shape, enc_outputs.shape))
    # 디코더는 룩어헤드 마스크(첫번째 서브층)과 패딩 마스크(두번째 서브층) 둘 다 사용.
    look_ahead_mask = Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = Input(shape=(1, 1, None), name='padding_mask')
    print('decoder : look_ahead_mask shape = {}'.format(look_ahead_mask.shape))
    print('decoder : padding_mask shape = {}'.format(padding_mask.shape))
    # 포지셔널 인코딩 + 드롭아웃
    embeddings = Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    print('decoder : before PE embeddings shape = {}'.format(embeddings.shape))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = Dropout(rate=dropout)(embeddings)

    # 디코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout,
                                name='decoder_layer_{}'.format(i))(
            inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
    return Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)


def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="transformer"):
    # 인코더 입력
    inputs = Input(shape=(None,), name="inputs")

    # 디코더 입력
    dec_inputs = Input(shape=(None,), name="dec_inputs")

    # 인코더의 패딩 마스크
    # Lambda Layer를 통해 create_padding_mask 메소드 기능을 수행하는 Layer를 바로 생성한다.
    enc_padding_mask = Lambda(create_padding_mask, output_shape=(1, 1, None),
                              name="enc_padding_mask")(inputs)
    print('Transformer : encoder enc_padding_mask shape = {}'.format(enc_padding_mask.shape))

    # 디코더의 룩어헤드 마스크(첫번째 서브층)
    look_ahead_mask = Lambda(create_look_ahead_mask, output_shape=(1, None, None),
                             name="look_ahead_mask")(dec_inputs)
    print('Transformer : decoder look_ahead_mask shape = {}'.format(look_ahead_mask.shape))

    # 디코더의 패딩 마스크(두번째 서브층)
    # 디코더라도 두번째 서브층에서 적용되는 Query인 디코더문장이 아니라, Key인 인코더 문장에 대해 패딩마스크를 씌워야함.
    dec_padding_mask = Lambda(create_padding_mask, output_shape=(1, 1, None),
                              name="dec_padding_mask")(inputs)
    print('Transformer : decoder dec_padding_mask shape = {}'.format(dec_padding_mask.shape))

    # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model,
                          num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_padding_mask])
    print('Transformer : encoder outputs shape = {}'.format(enc_outputs.shape))

    # 디코더의 출력은 dec_outputs. 출력층으로 전달.
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model,
                          num_heads=num_heads, dropout=dropout)(
        inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])
    print('Transformer : decoder outputs shape = {}'.format(dec_outputs.shape))

    # 다음 단어 예측을 위한 출력층
    outputs = Dense(units=vocab_size, name="outputs")(dec_outputs)
    print('Transformer : predict output shape = {}'.format(outputs.shape))

    return Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    # get_config 사용하고 싶지 않으면 model.save_weight와 load_weights 사용.
    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({
    #         'd_model': self.d_model,
    #         'warmup_steps': self.warmup_steps
    #     })
    #     return config

    # 왜 되는거야??...
    # def get_config(self):
    #     config = {
    #         # "d_model": self.d_model,
    #         "warmup_steps": self.warmup_steps
    #     }
    #
    #     return config

    def get_config(self):
        config = {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps
        }
        base_config = super(CustomSchedule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PositionalEncoding(Layer):
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

        print('pos_encoding shape = {}'.format(pos_encoding.shape))
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        # inputs shape는 보통 (batch_size, input_length(position), embeddingDim(d_model)) 일텐데
        # 왜 input length에 해당하는 부분에 axis를 새로 만드는걸까? inputs shape에 대해 다시 확인 필요.
        # -> 새로 만드는게 아니라 input_length 만큼 자름
        # print('pos_encoding call : inputs shape = {}, pos_encoding reshape = {}'.format(inputs.shape,
        #                                                                                 self.pos_encoding[:,
        #                                                                                 :tf.shape(inputs)[1], :].shape))

        # vocab_size,embedding_dim 만큼의 포지셔널 인코딩 벡터 생성 후, input_length만큼 position 값을 자름.
        # position 값이 변한다고 기존 포지셔널 인코딩 벡터값이 변하는 것이 아니므로 자르고 이어붙이기 가능.
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    1. WQ,WK,WV에 해당하는 d_model 크기의 밀집층(Dense Layer)을 지나게 한다.
    2. 지정된 헤드수(num_heads)만큼 나눈다(split)
    3. 스케일드 닷 프로덕트 어텐션
    4. 나눠졌던 헤드들을 연결(concatenate)한다
    5. WO에 해당하는 밀집층을 지나게 한다.
    """

    # 트랜스포머는 LSTM,RNN 같은 특정한 층으로부터 나온 은닉상태값이 없고 sequence_length 전체를 한꺼번에 관리가능하다.

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        # assert 테스트 : False이면 AssertionError
        assert d_model % self.num_heads == 0, 'd_model/num_heads 가 정수가 아님'

        # d_model을 num_heads로 나눈값
        # 논문 기준 : 64
        self.depth = d_model // self.num_heads

        # WQ, WK, WV에 해당하는 밀집층 정의
        # 가중치 행렬 크기 : (1 , d_model, d_model)?
        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)

        # WO에 해당하는 밀집층 정의

        self.dense = Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        # inputs 크기 : (batch_size, sequence_length, d_model)
        # 1. d_model 차원을 num_heads만큼 나눠야 하므로 (batch_size, sequence_length, num_heads, d_model/num_heads)로 만든다.
        # 2. num_heads 단위로 묶어진 각각의 행렬 수만큼 scaled dot product attention 계산해야하므로 num_heads 차원을 앞으로 뺀다.
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        # 참고) 인코더(k,v)-디코더(q) 어텐션에서는 query의 길이와 key, value의 길이는 다를 수 있다.
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        """num_heads만큼 다양한 시각을 가지긴 하지만, 그만큼 각각의 시각에서 가지는 capacity는 줄어들긴함"""

        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # 다시 헤드 연결하기 위해 num_heads와 d_model/num_heads를 이어놓기.
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, d_model)
        """ batch_size 굳이 명시해야하나? batch_size도 -1로하면 안되는건지...???? 테스트해보기 
            -1로 해도 되긴 한데.. 명시적으로 적어주는것도 나쁘진 않을지도..?"""
        # print('MultiHeadAttention : scaled_attention shape = {}'.format(scaled_attention.shape))
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # print('MultiHeadAttention : concat_attention shape = {}'.format(concat_attention.shape))

        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)

        return outputs


class TestPosEncoding:
    def process(self):
        input_length = 500
        sample_pos_encoding = PositionalEncoding(9000, 150)
        print('sample_pos_encoding shape = {}'.format(sample_pos_encoding.pos_encoding.shape))
        plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, 128))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()
        sample_pos_encoding.pos_encoding = sample_pos_encoding.pos_encoding[:, :500, :]
        print('sample_pos_encoding shape = {}'.format(sample_pos_encoding.pos_encoding.shape))
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
        # temp_k = tf.constant([[10, 0, 0],
        #                       [0, 10, 0],
        #                       [0, 0, 10],
        #                       [0, 0, 10]], dtype=tf.float32)  # (4, 3)
        #
        # temp_v = tf.constant([[1, 0],
        #                       [10, 0],
        #                       [100, 5],
        #                       [1000, 6]], dtype=tf.float32)  # (4, 2)

        temp_k = tf.constant([[10, 0, 0],
                              [0, 10, 0]], dtype=tf.float32)  # (2, 3)

        temp_v = tf.constant([[1, 0],
                              [10, 0]], dtype=tf.float32)  # (4, 2)

        temp_q = tf.constant([[10, 10, 0], [0, 10, 0]], dtype=tf.float32)  # (2, 3)

        temp_mask = 1 - tf.linalg.band_part(tf.ones((2, 2)), -1, 0)

        # 함수 실행
        temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, temp_mask)

        # 1. q와 k의 유사도를 구하는데, k의 행을 스캔하면서 단어별 유사도(어텐션 분포) 계산
        # 2. 구한 어텐션스코어를 가지고 v에 대입해 최종 어텐션 스코어 계산
        # 3. 최종적으로 어텐션 스코어의 shape는 (query의 문장 길이, value의 임베딩벡터(d_model) OR d_model/num_heads)

        # temp_q가 [[10, 10, 0], [0, 10, 0]] 일때
        # q의 첫번째 단어는 k의 첫번째와 두번째 단어와 어텐션 분포 유효값을 가지고 그 결과 v의 첫번쨰 두번쨰 상태를 참조하여 계산
        # q의 두번째 단어는 k의 두번째 단어와만 어텐션 분포 유효값을 가지고 있으며, 그 결과 v의 두번째 상태만을 가지고옴.

        print(temp_attn)  # 어텐션 분포(어텐션 가중치의 나열)
        print(temp_out)  # 어텐션 값


class TestMakeTransformer:
    def process(self):
        small_transformer = transformer(
            vocab_size=9000,
            num_layers=4,
            dff=512,
            d_model=128,
            num_heads=4,
            dropout=0.3,
            name="small_transformer"
        )
        plot_model(small_transformer, to_file='small_transformer.png', show_shapes=True)


if __name__ == "__main__":
    TestScaledDocProductAttention().process()
    # TestPosEncoding().process()
    # TestMakeTransformer().process()
