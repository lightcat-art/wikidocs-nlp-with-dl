import numpy as np


class Tensor:

    def __init__(self):
        pass

    def zeroDim(self):
        d = np.array(5)

        print('텐서의 차원 : ', d.ndim)  # ndim = 축(axis)의 개수, 텐서의 차원
        print('텐서의 크기(shape) : ', d.shape)

    def oneDim(self):
        d = np.array([1, 2, 3, 4])  # 1차원 텐서 = 벡터

        print('텐서의 차원 : ', d.ndim)  # ndim = 축(axis)의 개수, 텐서의 차원
        print('텐서의 크기(shape) : ', d.shape)

    def twoDim(self):
        d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])  # 2차원텐서= 행렬
        print('텐서의 차원 : ', d.ndim)  # ndim = 축(axis)의 개수, 텐서의 차원
        print('텐서의 크기(shape) : ', d.shape)

    def threeDim(self):
        d = np.array([[[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [6, 7, 8, 9, 10]],
                      [[1, 2, 3, 4, 5], [1, 4, 5, 6, 7], [12, 4, 22, 1, 4]]])
        print('텐서의 차원 : ', d.ndim)  # ndim = 축(axis)의 개수, 텐서의 차원
        print('텐서의 크기(shape) : ', d.shape)


if __name__ == "__main__":
    Tensor().zeroDim()
    Tensor().oneDim()
    Tensor().twoDim()
    Tensor().threeDim()
