import numpy as np
from numpy.linalg import norm


class Euclidean:

    # 두 원소간 거리를 구하는것
    def dist(self, x, y):
        return np.sqrt(np.sum(x - y) ** 2)

    # 특정 원소의 크기를 구하는것 -> 유클리디안거리와 상관없음
    def norm2(self, x, y):
        return norm(x)

    def test(self):
        doc1 = np.array([2, 3, 0, 1])
        doc2 = np.array([1, 4, 5, 6])
        doc3 = np.array([1, 7, 9, 110])
        docQ = np.array([1, 1, 0, 1])

        print('문서1과 문서Q 거리 : ', self.dist(doc1, docQ))
        print('문서2과 문서Q 거리 : ', self.dist(doc2, docQ))
        print('문서3과 문서Q 거리 : ', self.dist(doc3, docQ))


class Jaccard:
    def similarity(self, x, y):
        union = set(x).union(set(y))
        print('두 문서의 합집합 : ', union)

        intersection = set(x).intersection(set(y))
        print('두 문서의 교집합 : ', intersection)

        #두문서에서 추출된 단어토큰들중 두 문서가 겹치는 단어가 얼마나 되는지를 유사도로 표현
        return len(intersection) / len(union)

    def test(self):
        doc1 ="apple banana everyone like likey watch card holder"
        doc2 = "apple banana coupon passport love you"

        '''
        split 함수 설명
        If sep is not specified or is None, any
        whitespace string is a separator and empty strings are
        removed from the result.
        '''
        # 토큰화
        tokenize_doc1 = doc1.split()
        tokenize_doc2 = doc2.split()

        print('문서1 : ',tokenize_doc1)
        print('문서2 : ',tokenize_doc2)

        print(self.similarity(tokenize_doc1,tokenize_doc2))


if __name__ == "__main__":
    Jaccard().test()
