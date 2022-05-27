import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM

import pandas as pd
from string import punctuation


class ManyToOneRnn:
    def __init__(self):
        self.text = """경마장에 있는 말이 뛰고 있다\n그의 말이 법이다\n가는 말이 고와야 오는 말이 곱다"""
        self.max_len = None

    def build(self):
        ########## PREPROCESSING ############
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([self.text])
        vocab_size = len(tokenizer.word_index) + 1

        sequences = list()
        for line in self.text.split('\n'):  # 줄바꿈 문자를 기준으로 문장 토큰화
            # print(line)
            # print('1 : ',tokenizer.texts_to_sequences([line]))
            encoded = tokenizer.texts_to_sequences([line])[0]
            # print(encoded)
            for i in range(1, len(encoded)):
                sequence = encoded[:i + 1]  # 입력데이터와 레이블 데이터를 일단 같이 저장. 나중에 자르기.
                # print(sequence)
                sequences.append(sequence)

        # print(sequences)
        # print([len(l) for l in sequences])
        self.max_len = max(len(l) for l in sequences)
        print('샘플 최대 길이 : {}'.format(self.max_len))
        sequences = pad_sequences(sequences, maxlen=self.max_len, padding='pre')
        print(sequences)

        sequences = np.array(sequences)
        # -1 인덱스는 뒤에서 첫번째 인덱스를 의미, -2는 뒤에서 두번째 인덱스를 의미
        X = sequences[:, :-1]  # 열인덱스를 처음부터 뒤에서 첫번째인덱스 이전(-2)까지 슬라이스
        y = sequences[:, -1]

        y = to_categorical(y, num_classes=vocab_size)

        print(X)
        print(y)

        ########### MAKE MODEL ###########
        embedding_dim = 10
        hidden_units = 32

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(SimpleRNN(hidden_units))
        model.add(Dense(vocab_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # keras : X가 원핫인코딩이 되지 않고 그냥 들어가도 학습됨.
        # Embedding 내부에서 단어들의 정수값이 [8,4] 이라면 [8,4] 번째에 해당하는 가중치행렬의 행벡터들이 학습되도록 하면 되니까?
        model.fit(X, y, epochs=200, verbose=2)

        ########### PREDICT ##############
        # 다대일 구조라서 현재는 4번을 반복해야 문장예측이 완성되지만, 다대다 구조에서는 한번만 예측을 하면 문장예측이 되는 구조일듯
        print(self.sentence_generation(model, tokenizer, '경마장에', 4))
        print(self.sentence_generation(model, tokenizer, '그의', 2))
        print(self.sentence_generation(model, tokenizer, '가는', 5))

    def sentence_generation(self, model, tokenizer, current_word, n):  # 모델, 토크나이저, 현재 단어, 반복 횟수
        init_word = current_word
        sentence = ''

        # n번 반복
        for _ in range(n):
            # 현재 단어에 대한 정수 인코딩과 패딩
            encoded = tokenizer.texts_to_sequences([current_word])[0]
            # maxlen 달라져도(커져도) 예측가능한가? 예측은 가능한데 학습된 RNN의 time_steps와 다르게 입력이 들어가므로 예측이 약간 이상함.
            encoded = pad_sequences([encoded], maxlen=self.max_len-1, padding='pre')

            result = model.predict(encoded, verbose=0)
            result = np.argmax(result, axis=1)  # 가장 높은 점수를 가진 인덱스 추출
            for word, index in tokenizer.word_index.items():
                # 예측한 단어를 단어집합에서 추출
                if index == result:
                    break

            # 현재단어와 예측된 단어를 모두 다음 예측의 입력단어로 사용하기 위해 재구성
            current_word = current_word + ' ' + word

            # 예측단어를 문장에 저장
            sentence = sentence + ' ' + word

        sentence = init_word + sentence
        return sentence

    def buildLstm(self):
        ########## PREPROCESSING ###########
        df = pd.read_csv(
            'C:/Users/KDH/PycharmProjects/NLPStudy/resources/archive_nytimes_articles/ArticlesApril2018.csv')
        print(df.head())
        print(df.keys())

        # print(df['headline'].isnull().values.any())

        headline = []
        # append는 데이터전체를 append.
        # extend는 java의 list.addAll 이라고 생각하면 됨. 가장 바깥쪽 iterable 내 모든 항목을 넣음.
        headline.extend(list(df.headline.values))
        # print(len(headline))
        # print(headline[:5])

        headline = [word for word in headline if word != "Unknown"]
        # print(len(headline))
        # print(headline[:5])

        preprocessed_headline = [self.repreprocessing(x) for x in headline]
        # print(preprocessed_headline[:5])

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(preprocessed_headline)
        vocab_size = len(tokenizer.word_index) + 1

        sequences = list()
        for sentence in preprocessed_headline:
            # 각 샘플에 대한 정수 인코딩
            encoded = tokenizer.texts_to_sequences([sentence])[0]
            for i in range(1, len(encoded)):
                sequence = encoded[:i + 1]
                sequences.append(sequence)

        print(sequences[:12])

        index_to_word = {}
        for key, value in tokenizer.word_index.items(): # 인덱스를 단어로 바꾸기 위함
            index_to_word[value] = key

        print('빈도수 상위 500번 단어 : {}'.format(index_to_word[500]))

        self.max_len = max(len(l) for l in sequences)
        print('샘플 최대 길이 : {}'.format(self.max_len))
        sequences = pad_sequences(sequences, maxlen=self.max_len, padding='pre')
        print(sequences[:3])

        sequences = np.array(sequences)
        # -1 인덱스는 뒤에서 첫번째 인덱스를 의미, -2는 뒤에서 두번째 인덱스를 의미
        X = sequences[:, :-1]  # 열인덱스를 처음부터 뒤에서 첫번째인덱스 이전(-2)까지 슬라이스
        y = sequences[:, -1]

        y = to_categorical(y, num_classes=vocab_size)

        print(X[:3])
        print(y[:3])

        ########### MAKE MODEL ###########
        embedding_dim = 10
        hidden_units = 32

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(LSTM(hidden_units))
        model.add(Dense(vocab_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # keras : X가 원핫인코딩이 되지 않고 그냥 들어가도 학습됨.
        # Embedding 내부에서 단어들의 정수값이 [8,4] 이라면 [8,4] 번째에 해당하는 가중치행렬의 행벡터들이 학습되도록 하면 되니까?
        model.fit(X, y, epochs=200, verbose=2)

        ########### PREDICT ##############
        print(self.sentence_generation(model, tokenizer, 'i', 10))
        print(self.sentence_generation(model, tokenizer, 'how', 10))

    def repreprocessing(self, raw_sentence):
        # character 문자 단위로 쪼개기.
        preprocessed_sentence = raw_sentence.encode('utf8').decode('ascii', 'ignore')
        # print(preprocessed_sentence)
        # print(len(preprocessed_sentence))
        # 구두점 제거와 동시에 소문자화
        return ''.join(word for word in preprocessed_sentence if word not in punctuation).lower()


if __name__ == '__main__':
    # ManyToOneRnn().build()
    ManyToOneRnn().buildLstm()
