import numpy as np
import urllib.request

from tensorflow.keras.utils import to_categorical

class charRNN:
    def __init__(self):
        pass

    def build(self):
        ########## PREPROCESSING ###########

        # urllib.request.urlretrieve("http://www.gutenberg.org/files/11/11-0.txt", filename="11-0.txt")
        f = open('11-0.txt','rb')
        sentences = []
        for sentence in f:
            sentence = sentence.strip() # \r, \n 제거
            sentence = sentence.lower() # 소문자화
            sentence = sentence.decode('ascii','ignore')
            if len(sentence) > 0:
                sentences.append(sentence)

        f.close()

        print(sentences[:5])

        # 띄어쓰기를 구분자로하여 모두 list내 항목 모두 join
        total_data=' '.join(sentences)
        print('문자열의 길이 또는 총 문자의 개수 : %d' % len(total_data))

        print(type(total_data))
        print(set(total_data)) # string을 set으로 형변환하면 중복제거된 총사용된 문자집합이 생성됨.
        char_vocab = sorted(list(set(total_data)))

        # 문자에 고유한 정수 부여
        char_to_index = dict((char, index) for index, char in enumerate(char_vocab))
        print('문자 집합 : ',char_to_index)

        # 정수로부터 문자를 리턴하는 딕셔너리 구성
        index_to_char = {}
        for key, value in char_to_index.items():
            index_to_char[value] = key

        # appl(입력 시퀀스) -> pple(예측해야하는 시퀀스)
        # train_X = 'appl'
        # train_Y = 'pple'

        # 정한 문자길이만큼 문자열전체를 등분
        seq_length = 60
        n_samples = int(np.floor((len(total_data)-1) / seq_length))
        print('샘플의 수 : {}'.format(n_samples))


        train_X = []
        train_y = []
        for i in range(n_samples):
            X_sample = total_data[i * seq_length: (i+1) * seq_length]



if __name__=="__main__":
    charRNN().build()
