from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Sequential, load_model


class preprocessing:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = Sequential()

    def intEncoding(self):
        train_text = "The earth is an awesome place alive"
        sub_text = "The earth is an great place live"
        # 단어집합 생성
        self.tokenizer.fit_on_texts([train_text])
        # 정수 인코딩
        sequences = self.tokenizer.texts_to_sequences([sub_text])[0]

        # 단어 집합에 없는 단어는 빼고 정수 인코딩 결과 나옴.
        print("정수 인코딩 : ", sequences)
        print("단어 집합 : ", self.tokenizer.word_index)

    def padSeq(self):
        # padding : pre = 배열의 앞쪽이 대상. post = 배열의 뒤쪽이 대상
        # maxlen = 정해준 길이보다 길이가 긴 샘플은 값을 자르고, 짧은 샘플은 0으로 값 채우기.
        array = pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7]], maxlen=3, padding='pre')
        print(array)

    def wordEmbedding(self):
        # 1. 토큰화
        tokenized_test = [['Hope', 'to', 'see', 'you', 'soon'], ['Nice', 'to', 'see', 'you', 'again']]

        # 2. 각 단어에 대한 정수 인코딩
        self.tokenizer.fit_on_texts(tokenized_test)
        encode_text = self.tokenizer.texts_to_sequences(tokenized_test)

        # 3. 위 정수 인코딩 데이터가 아래의 임베딩 층의 입력이 된다.
        vocab_size = len(self.tokenizer.word_index)
        print(vocab_size)
        embedding_dim = 2

        self.model.add(Embedding(vocab_size, embedding_dim, input_length=5))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        print(self.model.summary())  # Embedding 층은 bias 파라미터를 사용하지 않음.

        self.model.save("model_name.h5")
        # modelLoad = load_model("model_name.h5")
        # modelLoad.summary()


class loadModel:
    def __init__(self):
        self.model = None

    def loadModel(self, modelName):
        self.model = load_model(modelName)

    def summary(self):
        self.model.summary()


if __name__ == "__main__":
    # preprocessing().intEncoding()
    # preprocessing().padSeq()
    preprocessing().wordEmbedding()
    l = loadModel()
    l.loadModel('model_name.h5213')
    print("summary Info~~~~~~~~~~~~~~~~~")
    l.summary()
