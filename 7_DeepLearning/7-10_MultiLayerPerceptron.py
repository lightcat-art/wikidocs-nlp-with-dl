import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input


class test:
    def __init__(self):
        self.vocab_size = None
        self.num_classes = None
        pass

    def textToMat(self):
        texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 너무 좋아요']

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        print(tokenizer.word_index)

        # DTM 행렬을 생성하는 count옵션
        print(tokenizer.texts_to_matrix(texts, mode='count'))
        # DTM 행렬과 비슷하지만 단어의 존재여부만 보는 binary
        print(tokenizer.texts_to_matrix(texts, mode='binary'))
        # tfidf 계산
        print(tokenizer.texts_to_matrix(texts, mode='tfidf'))
        # 각문서별 해당단어가 등장한 횟수/각문서별 등장한 모든 단어 개수
        print(tokenizer.texts_to_matrix(texts, mode='freq'))

    def news(self):
        newsdata = fetch_20newsgroups(subset='train')  # 훈련데이터만 리턴
        # newsdata = fetch_20newsgroups(subset='test') # 테스트데이터만 리턴
        # newsdata = fetch_20newsgroups(subset='all') # 모든 데이터 리턴

        # newsdata는 이미 레이블인코딩과 레이블명, 등에 대해 전처리가 어느정도 되어있는 객체임
        # print(newsdata.keys())
        # print(newsdata['data'][:5])
        # print(newsdata.data[:5])
        # print('객체내 속성들 : ', dir(newsdata))
        # print('객체타입 : ', type(newsdata))
        print('훈련용 샘플의 개수 : ', len(newsdata.data))
        # print('주제의 개수 : {}'.format(len(newsdata.target_names)))
        # print(newsdata.target_names)

        # data로부터 DataFrame 생성
        data = pd.DataFrame(newsdata.data, columns=['email'])
        # data에 target열 추가
        data['target'] = pd.Series(newsdata.target)

        data['target_names'] = pd.Series([newsdata.target_names[i] for i in newsdata.target])

        print('상위 5개 행 : \n', data[:5])
        print('\n')
        data.info()  # 메소드 자체에 출력기능이 있음.

        # null값 가진 샘플이 있는지
        print('null값 존재여부 : ', data.isnull().values.any())

        print('중복을 제외한 샘플의 수 : {}'.format(data['email'].nunique()))
        print('중복을 제외한 주제의 수 : {}'.format(data['target'].nunique()))

        data['target_names'].value_counts().plot(kind='bar')  # 이름으로 plot
        # data['target'].value_counts().plot(kind='bar') # 레이블 정수로 plot
        plt.show()

        # count에 대한 열은 열명이 없으므로 count 붙여주고, target 열은 drop=False 이므로 보존하여 새로운 열로 추가.
        print(data.groupby('target').size().reset_index(name='count'))

        newsdata_test = fetch_20newsgroups(subset='test', shuffle=True)
        train_email = data['email']
        train_label = data['target']
        test_email = newsdata_test.data  # list 형
        test_label = newsdata_test.target  # numpy.ndarray 형

        self.vocab_size = 10000
        self.num_classes = 20

        # X_train, X_test, index_to_word = self.prepare_data(train_email, test_email, 'binary')
        y_train = to_categorical(train_label,
                                 self.num_classes)  # label의 정수가 연속으로 매겨진 정수들이라면 num_classes 쓰지 않아도 자동으로 처리됨.
        y_test = to_categorical(test_label, self.num_classes)

        # index_to_word 데이터는 상위 10000개만 나오는게 아니라 모든단어에 대해 다 나옴. index_to_word 인덱스는 단어 정수 인코딩을 따라 1부터 시작.
        # print('빈도수 상위 1번쨰 단어 : {}'.format(index_to_word[1]))
        # print('빈도수 상위 9999번쨰 단어 : {}'.format(index_to_word[9999]))

        modes = ['binary', 'count','tfidf','freq']
        for mode in modes: # 4개의 모드에 대해 각각 아래 작업 반복
            X_train, X_test, _ = self.prepare_data(train_email, test_email, mode)
            loss, score = self.fit_and_evaluate(X_train, X_test, y_train, y_test)
            print(mode+' 모드의 테스트 정확도:', score, ' , 손실:',loss)


    def fit_and_evaluate(self, X_train, X_test, y_train, y_test):
        inputs = Input(shape=(self.vocab_size))
        x = Dense(256, activation='relu')(inputs)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # verbose
        # 0 = silent (과정 출력이 되지 않음)
        # 1 = progress bar까지 모두 출력
        # 2 = one line per epoch. loss와 metrics만 출력
        model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=0, validation_split=0.1)

        score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)

        return score[0], score[1]

    def prepare_data(self, train_data, test_data, mode):
        tokenizer = Tokenizer(num_words=self.vocab_size)  # vocab_size개수만큼만 사용하여 정수인코딩
        tokenizer.fit_on_texts(train_data)
        X_train = tokenizer.texts_to_matrix(train_data, mode=mode)  # 샘플수 x vocab_size 크기 행렬
        X_test = tokenizer.texts_to_matrix(test_data, mode=mode)  # 샘플수 x vocab_size 크기 행렬
        return X_train, X_test, tokenizer.index_word


if __name__ == "__main__":
    # test().textToMat()
    test().news()
