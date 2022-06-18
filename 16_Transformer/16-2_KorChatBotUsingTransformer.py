from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import transformer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow import keras
from tensorflow.keras.models import save_model, load_model


class KorChatBot:
    def __init__(self):
        self.train_data = None
        self.tokenizer = None
        self.START_TOKEN = None
        self.END_TOKEN = None
        self.VOCAB_SIZE = None
        self.MAX_LENGTH = 40  # tokenizer 패딩 최대 길이 40
        self.D_MODEL = 256
        self.NUM_LAYERS = 2
        self.NUM_HEADS = 8
        self.DFF = 512
        self.DROPOUT = 0.1
        self.model = None

        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        self.dataset = None
        self.EPOCHS = 100

    def preprocessing(self):
        urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
                                   filename="ChatBotData.csv")
        self.train_data = pd.read_csv('ChatBotData.csv')
        print(self.train_data.head())
        print('챗봇 샘플의 개수 : ', len(self.train_data))
        print('결측값 확인 : \n', self.train_data.isnull().sum())

        questions = []
        for sentence in self.train_data['Q']:
            # 구두점에 대해 띄어쓰기
            # ex) 12시 땡! -> 12시 땡 !
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            questions.append(sentence)

        answers = []
        for sentence in self.train_data['A']:
            # 구두점에 대해 띄어쓰기
            # ex) 12시 땡! -> 12시 땡 !
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            answers.append(sentence)

        # 서브워드텍스트인코더를 사용하여 질문, 답변 데이터로부터 단어 집합(vocabulary) 생성
        # questions와 answers 데이터 concat 하여 단어집합 생성
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers,
                                                                                   target_vocab_size=2 ** 13  # 8192
                                                                                   )

        # 시작 토큰과 종료 토큰에 대한 정수 부여.
        self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]

        # 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 +2
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2

        # 서브워드텍스트인코더 토크나이저의 .encode()를 사용하여 텍스트 시퀀스를 정수 시퀀스로 변환
        questions, answers = self.tokenize_and_filter(questions, answers)
        print('질문 데이터의 크기(shape) : {}'.format(questions.shape))
        print('답변 데이터의 크기(shape) : {}'.format(answers.shape))
        print('0번째 질문 = {}'.format(questions[0]))
        print('0번째 답변 = {}'.format(answers[0]))

        # 인코더와 디코더의 입력 그리고 레이블 만들기
        # 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다
        self.dataset = tf.data.Dataset.from_tensor_slices(({
                                                               'inputs': questions,
                                                               'dec_inputs': answers[:, :-1]
                                                               # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
                                                           }, {
                                                               'outputs': answers[:, 1:]
                                                               # 디코더의 출력 . 맨 처음 토큰(시작 토큰) 이 제거된다.
                                                           }))

        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.shuffle(self.BUFFER_SIZE)
        self.dataset = self.dataset.batch(self.BATCH_SIZE)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def makeModel(self):
        # keras 메모리 클리어
        tf.keras.backend.clear_session()

        self.model = transformer.transformer(
            vocab_size=self.VOCAB_SIZE,
            num_layers=self.NUM_LAYERS,
            dff=self.DFF,
            d_model=self.D_MODEL,
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT
        )

        # learning rate 는 상수여도 되고, LearningRateSchedule를 상속하여 만든 스케쥴러여도 됨.
        # 스케줄러는 초기화를 제외하면 학습시에 step이 변수가 되어 동작.
        learning_rate = transformer.CustomSchedule(self.D_MODEL)

        optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=[self.accuracy])
        self.model.fit(self.dataset, epochs=self.EPOCHS)

    # 챗봇 평가시에 수행할 전처리. 학습데이터 전처리와 동일.
    def preprocess_sentence(self, sentence):
        # 구두점에 대해 띄어쓰기
        # ex) 12시 땡! -> 12시 땡 !
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        return sentence

    def evaluate(self, sentence):
        # 입력 문장에 대한 전처리
        sentence = self.preprocess_sentence(sentence)

        # 입력 문장에 시작 토큰, 종료 토큰 추가
        # 디코더 인풋과 아웃풋인 답변(answer)에 대해서만 시작과 종료 토큰을 하나씩 제거한다. 질문(question)에 대해서는 하지 않음.
        sentence = tf.expand_dims(self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)

        # 디코더의 입력으로 사용되어, 나온 모델의 출력을 이어붙여, 그 다음 예측의 디코더의 입력으로 사용. 이것이 반복되는 방식.
        output = tf.expand_dims(self.START_TOKEN, 0)

        # 디코더의 예측 시작
        for i in range(self.MAX_LENGTH):
            predictions = self.model(inputs=[sentence, output], training=False)

            # 현재 시점의 예측 단어를 받아옴
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 만약 현재 시점의 예측 단어가 종료 토큰이라면 예측을 중단
            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break

            # 현재 시점의 예측 단어를 output(출력)에 연결.
            # output은 for문의 다음 루프에서 디코더의 입력이 된다.
            output = tf.concat([output, predicted_id], axis=-1)

        # 단어 예측이 모두 끝났다면 output을 리턴
        # squeeze : dimension size가 1인 axis에 대해 삭제처리.
        return tf.squeeze(output, axis=0)

    def predict(self, sentence):
        prediction = self.evaluate(sentence)
        # prediction == 디코더가 리턴한 챗봇의 대답에 해당하는 정수 시퀀스
        # tokenizer.decode()를 통해 정수 시퀀스를 문자열로 디코딩.
        # 혹시 i가 tokenizer.vocab_size보다 커지는 오류를 방지
        predicted_sentence = self.tokenizer.decode([i for i in prediction if i < self.tokenizer.vocab_size])

        print('Input : {}'.format(sentence))
        print('Output : {}'.format(predicted_sentence))

        return predicted_sentence

    def saveModel(self):
        # time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        # model_name = 'transformer_{}'.format(time_str)
        model_name = "transformer_EPOCHS_{}".format(self.EPOCHS)

        # save_model(self.model, model_name)
        self.model.save(model_name)

    def loadModel(self, model_name):
        # CustomSchedule. Please ensure this object is passed to the `custom_objects` argument. 에러로 인한 custom_objects 추가.
        self.model = load_model(model_name
                                , custom_objects={"CustomSchedule": transformer.CustomSchedule}
                                )

    def saveWeight(self):
        # time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        # model_name = 'transformer_{}'.format(time_str)
        model_name = "transformer_EPOCHS_{}".format(self.EPOCHS)

        # save_model(self.model, model_name)
        self.model.save_weights(model_name)

    def loadWeight(self, model_name):
        # keras 메모리 클리어
        tf.keras.backend.clear_session()

        self.model = transformer.transformer(
            vocab_size=self.VOCAB_SIZE,
            num_layers=self.NUM_LAYERS,
            dff=self.DFF,
            d_model=self.D_MODEL,
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT
        )
        self.model.load_weights(model_name)

    def accuracy(self, y_true, y_pred):
        # 레이블의 크기는 (batch_size, MAX_LENGTH -1)
        y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))
        return sparse_categorical_accuracy(y_true, y_pred)

    def loss_function(self, y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))

        loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def tokenize_and_filter(self, inputs, outputs):
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(inputs, outputs):
            # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
            sentence1 = self.START_TOKEN + self.tokenizer.encode(sentence1) + self.END_TOKEN
            sentence2 = self.START_TOKEN + self.tokenizer.encode(sentence2) + self.END_TOKEN
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

        # 패딩
        # padding='pre' or 'post'  : pre는 앞에, post는 뒤에 패딩.
        tokenized_inputs = pad_sequences(tokenized_inputs, maxlen=self.MAX_LENGTH, padding='post')
        tokenized_outputs = pad_sequences(tokenized_outputs, maxlen=self.MAX_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs


if __name__ == "__main__":
    TYPE = "EVAL_W"
    kc = KorChatBot()
    if TYPE == "LEARN_M":
        # saveModel
        kc.preprocessing()
        kc.makeModel()
        kc.saveModel()
    elif TYPE == "EVAL_M":
        # CustomSchedule 때문에 load_model이 제대로 동작하지 않는 이슈..
        model_name = "transformer_EPOCHS_{}".format(kc.EPOCHS)
        print(model_name)
        kc.preprocessing()
        kc.loadModel(model_name)

        sen_list = []
        sen_list.append("영화 보고 싶어요")
        sen_list.append("고민이 요즘 많아요")
        sen_list.append("행복하게 사는 방법이 있을까요?")
        sen_list.append("전문지식은 어떻게 구축해야하나요?")
        sen_list.append("커피 먹고 싶어요")
        for sentence in sen_list:
            output = kc.predict(sentence)

    elif TYPE == "LEARN_W":
        # saveModel
        kc.preprocessing()
        kc.makeModel()
        kc.saveWeight()
    elif TYPE == "EVAL_W":
        model_name = "transformer_EPOCHS_{}".format(kc.EPOCHS)
        print(model_name)
        kc.preprocessing()
        kc.loadWeight(model_name)

        sen_list = []
        sen_list.append("영화 볼래?")
        sen_list.append("고민이 있어")
        sen_list.append("너무 화가나")
        sen_list.append("카페갈래?")
        sen_list.append("게임하고싶당")
        for sentence in sen_list:
            output = kc.predict(sentence)
