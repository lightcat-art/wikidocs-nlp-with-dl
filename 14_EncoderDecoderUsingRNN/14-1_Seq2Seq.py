import os
import shutil
import zipfile

import keras.models
import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.util import nest


class seq2seqChar:
    def __init__(self):
        self.encoder_model = None
        self.decoder_model = None
        self.index_to_src = None
        self.index_to_tar = None
        self.src_vocab_size = None
        self.tar_vocab_size = None
        self.encoder_input = None
        self.decoder_input = None
        self.decoder_target = None
        self.max_src_len = None
        self.max_tar_len = None
        self.src_to_index = None
        self.tar_to_index = None
        # 1. encoder
        self.encoder_inputs = None
        self.encoder_states = None
        # 2. decoder
        self.decoder_inputs = None
        self.decoder_lstm = None
        self.decoder_softmax_layer = None
        self.lines = None
        pass

    def build(self):
        ########### PREPROCESSING ###############

        # http = urllib3.PoolManager()
        # url = 'http://www.manythings.org/anki/fra-eng.zip'
        # filename = 'fra-eng.zip'
        # path = os.getcwd()
        # zipfilename = os.path.join(path, filename)
        # # 읽어들이는 r과 기록할 out_file을 동시에 가져옴.
        # with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:
        #     # r의 내용을 out_file로 이동.
        #     shutil.copyfileobj(r, out_file)

        # with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
        #     # 압축풀기
        #     zip_ref.extractall(path)

        # fra.txt의 구분자가 \t 이므로 \t구분자로 읽어옴. 컬럼명도 지정.
        self.lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')

        # print(lines[:5])
        # lic 컬럼데이터 삭제
        del self.lines['lic']
        # print(lines[:5])
        # print(type(lines))
        # print('전체 샘플의 개수 : {}'.format(len(lines)))

        # del lines['lic']과 똑같은 기능을 하는것으로 보여지므로 생략.
        # lines = lines.loc[:, 'src':'tar']

        self.lines = self.lines[0:60000]
        # print(lines[:5])
        # 랜덤 샘플 출력.
        # print(lines.sample(10))

        # tar 컬럼 데이터에 <sos>인 \t와 <eos>인 \n 추가.
        self.lines.tar = self.lines.tar.apply(lambda x: '\t ' + x + ' \n')

        # print(lines.sample(10))

        # 문자 집합 구축
        src_vocab = set()
        for line in self.lines.src:  # 1줄씩 읽음
            for char in line:  # 1개 문자씩 읽음
                src_vocab.add(char)

        tar_vocab = set()
        for line in self.lines.tar:
            for char in line:
                tar_vocab.add(char)

        # 패딩토큰을 고려하여 +1
        self.src_vocab_size = len(src_vocab) + 1
        self.tar_vocab_size = len(tar_vocab) + 1

        # 인덱스를 활용하기 위한 형변환 및 정렬
        src_vocab = sorted(list(src_vocab))
        tar_vocab = sorted(list(tar_vocab))

        self.src_to_index = dict([(word, i + 1) for i, word in enumerate(src_vocab)])
        self.tar_to_index = dict([(word, i + 1) for i, word in enumerate(tar_vocab)])

        self.encoder_input = []

        # 1개의 문장
        for line in self.lines.src:
            encoded_line = []
            # 각 줄에서 1개의 char
            for char in line:
                # 각 char을 정수로 변환
                encoded_line.append(self.src_to_index[char])
            self.encoder_input.append(encoded_line)
        print('source 문장의 정수 인코딩 : ', self.encoder_input[:5])

        self.decoder_input = []

        for line in self.lines.tar:
            decoded_line = []
            for char in line:
                decoded_line.append(self.tar_to_index[char])
            self.decoder_input.append(decoded_line)

        print('target 문장의 정수 인코딩 : ', self.decoder_input[:5])

        self.decoder_target = []

        for line in self.lines.tar:
            timestep = 0
            encoded_line = []
            for char in line:
                if timestep > 0:
                    encoded_line.append(self.tar_to_index[char])
                timestep = timestep + 1
            self.decoder_target.append(encoded_line)

        self.max_src_len = max([len(line) for line in self.lines.src])
        self.max_tar_len = max([len(line) for line in self.lines.tar])

        # 문자단위 번역기이므로 워드임베딩은 별도로 사용x, 실제값뿐만 아니라 입력값도 원핫벡터 사용.
        self.encoder_input = pad_sequences(self.encoder_input, maxlen=self.max_src_len, padding='post')
        self.decoder_input = pad_sequences(self.decoder_input, maxlen=self.max_tar_len, padding='post')
        self.decoder_target = pad_sequences(self.decoder_target, maxlen=self.max_tar_len, padding='post')
        print('after pad sequence : encoder_input : [', len(self.encoder_input),',', len(self.encoder_input[0]),']')

        self.encoder_input = to_categorical(self.encoder_input)
        print('after to categorical : encoder_input : [', len(self.encoder_input), ',', len(self.encoder_input[0]), ',', len(self.encoder_input[0][0]),']')
        self.decoder_input = to_categorical(self.decoder_input)
        self.decoder_target = to_categorical(self.decoder_target)

        self.index_to_src = dict((i, char) for char, i in self.src_to_index.items())
        self.index_to_tar = dict((i, char) for char, i in self.tar_to_index.items())

    def createLearnModel(self):
        ############ Model create ##############

        # 1. encoder
        self.encoder_inputs = Input(shape=(None, self.src_vocab_size))
        print('model encoder inputs shape : ', self.encoder_inputs.shape)
        encoder_lstm = LSTM(units=256, return_state=True)  # many to one

        # encoder_outputs는 여기서는 불필요
        encoder_outputs, state_h, state_c = encoder_lstm(self.encoder_inputs)
        print('model encoder outputs shape : ', encoder_outputs.shape)

        # LSTM은 바닐라 RNN과는 달리 상태가 두개. 은닉 상태와 셀 상태
        self.encoder_states = [state_h, state_c]

        # 2. decoder
        self.decoder_inputs = Input(shape=(None, self.tar_vocab_size))
        self.decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

        # 디코더에게 인코더의 은닉상태, 셀상태 전달
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs, initial_state=self.encoder_states)

        self.decoder_softmax_layer = Dense(self.tar_vocab_size, activation='softmax')
        decoder_outputs = self.decoder_softmax_layer(decoder_outputs)

        model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        model.fit(x=[self.encoder_input, self.decoder_input], y=self.decoder_target, batch_size=64, epochs=1,
                  validation_split=0.2)

    def createPredictModel(self):
        ############# create predict Model ###########
        #### 1. encoder predict model
        self.encoder_model = Model(inputs=self.encoder_inputs, outputs=self.encoder_states)

        #### 2. decoder predict model
        decoder_state_input_h = Input(shape=(256,))
        decoder_state_input_c = Input(shape=(256,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # 문장의 다음단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용.
        decoder_outputs_pred, decoder_state_h, decoder_state_c = self.decoder_lstm(self.decoder_inputs,
                                                                                   initial_state=decoder_states_inputs)

        # 훈련 과정에서와 달리 LSTM의 리턴 은닉 상태와 셀 상태 버리지 않음
        decoder_states = [decoder_state_h, decoder_state_c]
        decoder_outputs_pred = self.decoder_softmax_layer(decoder_outputs_pred)

        self.decoder_model = Model(inputs=[self.decoder_inputs] + decoder_states_inputs,
                                   outputs=[decoder_outputs_pred] + decoder_states)

        self.encoder_model.save("encoder_model.h5")
        self.decoder_model.save("decoder_model.h5")

    def loadModel(self):
        self.encoder_model = keras.models.load_model("encoder_model.h5")
        self.decoder_model = keras.models.load_model("decoder_model.h5")

    def decode_sequence(self, input_seq):
        # 입력으로부터 인코더의 상태 얻기.
        states_value = self.encoder_model.predict(input_seq)

        # <SOS>에 해당하는 원-핫 벡터 생성
        target_seq = np.zeros((1, 1, self.tar_vocab_size))
        target_seq[0, 0, self.tar_to_index['\t']] = 1.

        # test = [target_seq]+states_value
        # print('target_seq + states_value = ', [target_seq] + states_value)
        # print(len(nest.flatten([target_seq] + states_value))) #3
        # print(isinstance([target_seq] + states_value, list)) # True
        # print(nest.is_nested([target_seq] + states_value)) # True
        # print(nest.map_structure(lambda x : x.shape , test)) # [(1, 1, 105), (1, 256), (1, 256)]
        # print('len of target_seq + states_value = ', len([target_seq] + states_value))
        # print('target_seq = ', target_seq)
        # print('states_value = ', states_value)


        stop_condition = False
        decoded_sentence = ""

        # stop_condition이 True가 될때까지 반복
        while not stop_condition:
            # 이전 시점의 상태 states_value를 현 시점의 초기상태로 사용

            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # 예측 결과를 문자로 변환
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.index_to_tar[sampled_token_index]

            # 현재 시점의 예측 문자를 예측 문장에 추가
            decoded_sentence += sampled_char

            # <eos>에 도달하거나 최대 길이를 넘으면 중단.
            if (sampled_char == '\n' or len(decoded_sentence) > self.max_tar_len):
                stop_condition = True

            # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장.
            target_seq = np.zeros((1, 1, self.tar_vocab_size))
            target_seq[0, 0, sampled_token_index] = 1.

            # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장.
            states_value = [h, c]

        return decoded_sentence

    def predict(self):
        for seq_index in [3, 50, 100, 300, 1001]:
            input_seq = self.encoder_input[seq_index:seq_index + 1]
            decoded_sentence = self.decode_sequence(input_seq)

            print(35 * "-")
            print('입력문장:', self.lines.src[seq_index])
            print('정답문장:', self.lines.tar[seq_index][2:len(self.lines.tar[seq_index]) - 1])  # '\t'와 '\n'을 빼고 출력
            print('번역문장:', decoded_sentence[1:len(decoded_sentence) - 1])  # '\n'을 빼고 출력


if __name__ == "__main__":
    s = seq2seqChar()

    # 학습된 모델 저장.
    s.build()
    s.createLearnModel()
    # s.createPredictModel()
    # s.predict()

    # 앞에서 model을 저장시키고 재사용가능
    # s.build()
    # s.loadModel()
    # s.predict()
