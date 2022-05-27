import numpy as np
from numpy import dot
from numpy.linalg import norm

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class COS_SIM_THEORY:
    def cos_sim(self, A, B):
        # 2-norm 을 사용
        return dot(A, B) / (norm(A) * norm(B))

    def test(self):
        doc1 = np.array([0, 1, 1, 1])
        doc2 = np.array([1, 0, 1, 1])
        doc3 = np.array([2, 0, 2, 2])

        print(norm(doc1))
        print(dot(doc1, doc2))
        print(dot(doc1, doc3))
        print(dot(doc2, doc3))

        print('문서 1과 문서 2의 유사도 : ', self.cos_sim(doc1, doc2))
        print('문서 1과 문서 3의 유사도 : ', self.cos_sim(doc1, doc3))
        print('문서 2과 문서 3의 유사도 : ', self.cos_sim(doc3, doc2))

        # doc_mat = np.matrix([[0,0,1,1],[1,1,1,1]])
        # print(doc_mat)
        # print(norm(doc_mat))


class RECOMM_USE_COS_SIM:

    def __init__(self):
        self.title_to_index = None
        self.data = None
        self.cosine_sim = None

    def recomm_system(self):
        self.data = pd.read_csv('C:/Users/KDH/PycharmProjects/NLPStudy/resources/archive_movie/movies_metadata.csv',
                                low_memory=False)
        self.data = self.data.head(20000)

        # overview 열에 존재하는 모든 결측값을 체크하고, 카운트.
        print('overview 열의 결측값의 수:', self.data['overview'].isnull().sum())

        # 결측값을 빈값으로 대체
        self.data['overview'] = self.data['overview'].fillna('')

        # english : 영어로 기본등록되어있는 불용어도 같이 제거됨.
        tfidf = TfidfVectorizer(stop_words='english')

        tfidf_matrix = tfidf.fit_transform(self.data['overview'])
        print('TF-IDF 행렬 크기 : ', tfidf_matrix.shape)

        # \ 방향의 대각선 행렬원소들은 자기자신문서에 대한 유사도이므로 모두 1
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print('코사인 유사도 연산 행렬 크기 : ', self.cosine_sim.shape)
        # print('코사인 유사도 연산 행렬  : ',self.cosine_sim)

        # zip : data['title']열과 data.index 를 매핑시켜 tuple list 리턴.
        self.title_to_index = dict(zip(self.data['title'], self.data.index))
        # print(title_to_index)

        idx = self.title_to_index['Father of the Bride Part II']
        print(idx)

        title = 'The Dark Knight Rises'
        # print(self.get_recommendations(title))
        print(self.get_recommendations(title))

    def get_recommendations(self, title):
        idx = self.title_to_index[title]

        # print(self.cosine_sim[idx])
        # print(self.cosine_sim[idx].shape)

        # 해당 영화와 모든 영화와의 유사도를 가져온다.
        # enumerate를 통해 가져온 코사인유사도리스트에 인덱스 부여한 tuple 반환, 코사인유사도 구하는과정에서 문서 idx가 어그러지지 않았으면 그대로 사용이 가능하므로.
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        # print('sim_score : \n',sim_scores)

        # sim_scores에서 2번쨰 열인 유사도점수 기준으로 DESC로 정렬
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # print('sim_score : \n', sim_scores)

        # 가장 유사한 10개의 영화 유사도스코어 받아오기.
        sim_scores = sim_scores[1:11]

        # 가장 유사한 10개의 영화 인덱스 가져오기.
        movie_indices = [idx[0] for idx in sim_scores]

        # 가장 유사한 10개의 영화 제목 리턴.
        return self.data['title'].iloc[movie_indices]


# explicitly assigned main class
if __name__ == "__main__":
    # COS_SIM_THEORY().test()
    RECOMM_USE_COS_SIM().recomm_system()
    pass
