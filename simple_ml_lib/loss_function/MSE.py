from typing import List
import numpy as np

def MSELoss(Y_pre:List[float],Y_real:List[float]):
    #均方误差损失
    if len(Y_pre)!=len(Y_real):
        raise ValueError("The length of two list should be the same")
    loss = np.mean((Y_pre - Y_real) ** 2) / 2
    return loss

