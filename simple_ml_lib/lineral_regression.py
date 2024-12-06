import numpy as np
from typing import List
from .loss_function.MSE import MSELoss

def LeastSquareMethod(X : List[List[float]], Y : List[List[float]]):

    #这里使用了列表套列表而不是np.array实现,具体内容可以看notes
    #判断X,Y是否长度相同

    if len(X)!=len(Y):
       raise ValueError("The length of two list should be the same")
    
    #把二者转换成numpy array的格式方便计算
    X = np.array(X)
    X = np.column_stack((np.ones(len(X)), X)) #根据需求看是否加上这一项
    Y = np.array(Y)

    #将X进行转置
    X_Transpose = X.T
    
    #计算 X^T X
    XTX = np.dot(X_Transpose, X)
    #计算 X^T Y
    XTY = np.dot(X_Transpose, Y)
    #计算 (X^T X)^{-1}
    XTX_inv = np.linalg.inv(XTX)
    #计算参数矩阵 β
    beta = np.dot(XTX_inv, XTY)

    return beta

'''
def GradientDescent(learning_rate : float, 
                    X : List[List[float]], 
                    Y : List[List[float]]):

    K_list = [[0] for i in range(len(X))]
'''    