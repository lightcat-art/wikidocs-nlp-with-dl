from nltk.tokenize import word_tokenize, WordPunctTokenizer, TreebankWordTokenizer, sent_tokenize
import nltk
from nltk.tag import pos_tag
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import kss
from konlpy.tag import Okt, Kkma


class Tokenization:

    def tokenize_test(self):
        # tokenize 별로 분리되는 규칙이 다르다.
        nltk.data.path.clear()
        nltk.data.path.append('C:\\Users\KDH\\anaconda3-64bit\\envs\\nlpEnv\\NLTK_DATA')
        print('단어 토큰화1 :', word_tokenize(
            "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
        print('단어 토큰화2 :', WordPunctTokenizer().tokenize(
            "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
        print('단어 토큰화3 :', text_to_word_sequence(
            "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

        # 1. 하이푼으로 구성된 단어는 하나로 유지한다.
        # 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.
        tokenizer = TreebankWordTokenizer()
        text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
        print('트리뱅크 워드토크나이저 :', tokenizer.tokenize(text))

    def sentence_tokenize(self):
        text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
        print('문장 토큰화1 : ', sent_tokenize(text))

        text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
        print('문장 토큰화2 : ', sent_tokenize(text))

        text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
        print('문장 토큰화3 : ', kss.split_sentences(text))

    # pos_tag : part of speech tagging
    def posTaggingEng(self):
        text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
        tokenized_sentence = word_tokenize(text)
        print('단어 토큰화 :', tokenized_sentence)
        # Penn Treebank POS Tags : PRP(인칭대명사), VBP(동사), RB(부사), VBG(현재부사), IN(전치사), NNP(고유명사), NNS(복수형), CC(접속사), DT(관사)
        print('품사 태깅 : ', pos_tag(tokenized_sentence))

    def posTaggingKor(self):
        okt = Okt()
        kkma = Kkma()

        print('OKT 형태소 분석 : ', okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
        print('OKT 품사 태깅 : ', okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
        print('OKT 명사 추출 : ', okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
        
        print('꼬꼬마 형태소 분석 : ', kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
        print('꼬꼬마 품사 태깅 : ', kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
        print('꼬꼬마 명사 추출 : ', kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
        



# explicitly assigned main class
if __name__ == "__main__":
    # Tokenization().tokenize_test()
    # Tokenization().sentence_tokenize()
    # Tokenization().posTaggingEng()
    Tokenization().posTaggingKor()
    pass
