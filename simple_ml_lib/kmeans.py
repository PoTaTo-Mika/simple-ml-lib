import random as rd
import numpy as np
from typing import List
import pandas as pd

def calculate_distance(X, Y):
    return np.sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)

def Kmeans(Points: List[List[float]], cluster: int, epoch: int):

    #这里我们用列表套列表表示矩阵，不用numpy库
    nums = len(Points)
    cluster_points = [Points[rd.randint(0, nums-1)] for _ in range(cluster)]
    
    for _ in range(epoch):
        #初始化簇
        clusters = [[] for _ in range(cluster)]
        
        #为每个点分配簇
        for point in Points:
            distances = [calculate_distance(point, cp) for cp in cluster_points]
            closest_cluster = distances.index(min(distances))
            clusters[closest_cluster].append(point)
        
        #更新簇中心
        for i in range(cluster):
            if clusters[i]:
                cluster_points[i] = [sum(coord)/len(clusters[i]) for coord in zip(*clusters[i])]
    
    return clusters, cluster_points

