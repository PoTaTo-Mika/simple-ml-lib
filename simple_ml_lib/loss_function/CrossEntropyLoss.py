import numpy as np
from typing import List

"""
交叉熵损失的计算要先对预测值进行softmax处理，然后再计算交叉熵损失，所以我们自己还要额外实现一个softmax函数。
"""

def softmax(x:List[float]):
    """
    计算softmax值
    """
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def CrossEntropy(p:List[float],q:List[float]):
    """
    P和Q都是概率分布，交叉熵代表q(x)相对于p(x)的不确定性，也就是困难程度。
    """
    p = np.array(p)
    q = np.array(q)

    return -np.sum(p*np.log(q))
    

def CrossEntropyLoss(y:List[float], y_hat:List[float]):
    """
    计算交叉熵损失
    """
    y = np.array(y)
    y_hat = np.array(y_hat)

    return CrossEntropy(y, y_hat)

if __name__ == "__main__":
    """
    Test
    """
    true_labels = ['1','4','5']
    predictions = [
    [0.1, 0.6, 0.3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.3, 0.2, 0, 0.5, 0, 0, 0, 0, 0],
    [0.6, 0.3, 0, 0, 0, 0.1, 0, 0, 0, 0]
    ]
    labels = ['0','1','2','3','4','5','6','7','8','9']
    Loss = CrossEntropyLoss(true_labels, predictions)
    print(Loss)