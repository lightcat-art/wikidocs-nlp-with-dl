import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding, Input, GlobalAveragePooling1D


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads  # 하나의 헤드에 존재하는 차원수
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)  # Head의 projection_dim
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)

        # output 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]

        # (batch_size, seq_len, embedding_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(query, key, value)

        # head 다시 concatenate
        # (batch_size, seq_len, num_heads, projection_dim)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len, embedding_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))

        outputs = self.dense(concat_attention)
        return outputs


class TransformerBlock(tf.keras.layers.Layer):
    """
    멀티 헤드 어텐션에 포지션 와이즈 피드 포워드 신경망을 추가한 인코더
    """

    def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(dff, activation="relu"),
             tf.keras.layers.Dense(embedding_dim)]
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)  # 첫번째 서브층 : 멀티 헤드 어텐션
        # training 옵션 : 학습시에만 적용되고 추론시에는 적용되지 않도록 하는 옵션
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & norm
        ffn_output = self.ffn(out1)  # 두번째 서브층 : 포지션 와이즈 피드 포워드 신경망
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & norm


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(vocab_size, embedding_dim)
        self.pos_emb = Embedding(max_len, embedding_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        # position 텐서 초기화
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class IMDBModel:
    def __init__(self):
        self.vocab_size = 20000  # 빈도수 상위 2만개 단어만 사용
        self.max_len = 200  # 문장 최대길이 200개만 사용
        self.embedding_dim = 32
        self.num_heads = 2
        self.dff = 32
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def preprocessing(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.imdb.load_data(
            num_words=self.vocab_size)
        print('훈련용 리뷰 개수 = {}'.format(len(self.X_train)))
        print('테스트용 리뷰 개수 = {}'.format(len(self.X_test)))

        self.X_train = tf.keras.preprocessing.sequence.pad_sequences(self.X_train, maxlen=self.max_len)
        self.X_test = tf.keras.preprocessing.sequence.pad_sequences(self.X_test, maxlen=self.max_len)

    def makeModel(self):
        inputs = Input(shape=(self.max_len,))
        embedding_layer = TokenAndPositionEmbedding(self.max_len, self.vocab_size, self.embedding_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(self.embedding_dim, self.num_heads, self.dff)
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)  # ??
        x = Dropout(0.1)(x)
        x = Dense(20, activation="relu")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(2, activation="softmax")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        history = self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=2,
                                 validation_data=(self.X_test, self.y_test))
        print("테스트 정확도 : %.4f" % (self.model.evaluate(self.X_test, self.y_test)[1]))


if __name__ == "__main__":
    imdb = IMDBModel()
    imdb.preprocessing()
    imdb.makeModel()
