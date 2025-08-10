import pandas as pd
# import scipy.io as scio
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
import math
from matplotlib import pyplot as plt
import math
# import getMap

# data = pd.read_csv("pathLoss.csv", header=None)
# data = pd.read_csv("Snr.csv", header=None)
# data = np.array(data).reshape(-1, 1)
# ac = np.load('radioenvir.npz') # 读取数据G:\GraPaper\王俊_paper\DJI\radioenvir1_dB100.npz
ac = np.load('results/datas/radioenvir_SINR-0.5_dB_100.npz') # 读取数据
data = ac['arr_0'].reshape(-1, 1) # map information # len= 40401
# ac_map = np.zeros(shape=(201,201))
#print(sum(data))

# data = ac_map.reshape(-1,1)
# print(len(data))
Min_Num = 10
Row = 201 # grid point # 网格点的数据
Col = 201
# valid = np.where(data > Min_Num)[0]
valid = np.where(data == 0)[0]  # 环境地图 # np.where()[0] 表示行索引，np.where()[1]表示列索引 返回索引 if Pout=1，valid=0
#print("vaild :",valid)
valid_map = {}

for t in range(valid.shape[0]):# t=[0,33817]
    valid_map[valid[t]] = t
print(valid.shape[0])
# 生成方向图
def getGraph():
    Graph = np.zeros(shape=(valid.shape[0], 8))
    Graph.fill(np.inf)
    for index in tqdm(range(valid.shape[0])):
        text = valid[index]
        row = text // Row
        col = text % Col
        index2 = 0
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if (i == row and j == col) or i < 0 or i >= Row or j < 0 or j >= Col:
                    continue
                # if data[i * Row + j] > Min_Num:
                if data[i * Row + j] == 0:
                    Graph[index][index2] = i * Row + j
                    index2 += 1
    return Graph

# 生成代价图
def getCost(Graph):
    Cost = np.zeros(shape=(Graph.shape[0], Graph.shape[1]))
    Cost.fill(np.inf)
    for index in tqdm(range(Graph.shape[0])):
        text = valid[index]
        row = text // Row
        col = text % Col
        for index1 in range(Graph.shape[1]):
            if Graph[index][index1] == np.inf:
                continue
            distance = Graph[index][index1]
            dis_row = distance // Row
            dis_col = distance % Col
            Cost[index][index1] = np.sqrt(np.square(row - dis_row) + np.square(col - dis_col))
            # index2 = int(Graph[index][index1])
            # Cost[index][index1] = np.sqrt(np.square(row - dis_row) + np.square(col - dis_col)) / np.log2((1 + data[index2]))
    return Cost


# 运用dijkstra
def dijkstra(Start, Cost, Graph, End):
    heap = []
    dis = np.zeros(shape=(Graph.shape[0], 2))
    vis = np.zeros(shape=(Graph.shape[0], 1))
    dis.fill(np.inf)
    Start_index = valid_map.get(Start)
    End_index = valid_map.get(End)
    dis[Start_index][0] = 0
    for index in range(Graph[Start_index].shape[0]):
        dot = Graph[Start_index][index]
        dot_index = valid_map.get(dot)
        dis[dot_index][0] = dis[Start_index][0] + Cost[Start_index][index]
        # dis[dot_index][0] = 1 / (1 / dis[dot_index][0] + Cost[dot_index][index]
        # print("dot_index",dot_index)
        dis[dot_index][1] = Start
        # print("dis[dot_index][1]",dis[dot_index][1])
        heapq.heappush(heap, (dis[dot_index][0], dot))
    while heap and vis[End_index] == 0:
        # while heap:
        vic = heapq.heappop(heap)
        dot_index = valid_map.get(vic[1])
        if vis[dot_index] == 1:
            continue
        vis[dot_index] = 1
        for index in range(Graph[dot_index].shape[0]):
            if Graph[dot_index][index] == np.inf:
                continue
            dot_next = Graph[dot_index][index]
            dot_next_index = valid_map.get(dot_next)
            # if vis[dot_next_index] == 0 and (dis[dot_next_index][0] > 1 / (1 / dis[dot_index][0] + Cost[
            # dot_index][ index])):
            if vis[dot_next_index] == 0 and (dis[dot_next_index][0] > dis[dot_index][0] + Cost[dot_index][index]):
                dis[dot_next_index][0] = dis[dot_index][0] + Cost[dot_index][index]
                # dis[dot_next_index][0] = 1 / (1 / dis[dot_index][0] + Cost[dot_index][index])
                dis[dot_next_index][1] = vic[1]
                heapq.heappush(heap, (dis[dot_next_index][0], dot_next))
    return dis


def get_path_len(path):
    """ 计算路径的长度 """
    path_length = 0
    # print(path)
    for i in range(1, len(path)):
        node1_x = path[i][0]
        node1_y = path[i][1]
        node2_x = path[i - 1][0]
        node2_y = path[i - 1][1]
        path_length += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)
    return path_length

# 生成路径
def getPath(Dji, Start, End):
    Path = []
    start_index = valid_map.get(Start)
    while 1:
        index = valid_map.get(End)
        if index == start_index:
            break
        Path.append(End) #
        #print(Dji)
        End = Dji[index][1]
    Path.append(Start)   # 添加起点位置
    # print(Path.shape)
    return Path
"""
引出接口，供SA函数调用，并保存数据
input  start location 
output end   location 


# """
def compt_dis_Dji(loc1,loc2):
    step = 201 #网格数目
    path_norm = []  # 存储路径
    graph = getGraph()
    cost = getCost(graph)
    # (y,x)位置数据转换成DJI算数数据结构
    start = loc1[1] * step + loc1[0]
    end  =  loc2[1] * step + loc2[0]# 1500,1100


    path = np.array(getPath(dijkstra(start, cost, graph, end), start, end))
    np.savetxt("path(Dji)"+str(loc1)+"_"+str(loc2)+".csv", path, delimiter=",") # 保存路径数据
    y_uav = path // 201 # 转换路径数据
    x_uav = path % 201
    for i in range(len(x_uav)):
        path_norm.append([x_uav[i], y_uav[i]])
    # print(path_norm)
    path_length = get_path_len(path_norm)  # 计算路径长度
    # print("path_length",path_length)
   # 计算路径长度
    return path_length
#===============================主函数
if __name__ == '__main__':
# 标准单位


    # loc1=[137,157]
    # loc2=[120,100]

    loc1 = [90, 90]
    loc2 = [110, 25]
# loc2=[120,32]
    length = compt_dis_Dji(loc1,loc2) # 注意用户位置的单位
"""
1 9.0 9.30
2 2.5 10.0
3 15.0 4.0
4 7.50 15.80
5 13.70 15.70
6 12.00 10.00
7 16.70 12.60
EOF
"""

def power(v):
    P0 = 79.8563
    Pi = 88.6279
    U_tip = 120
    v0 = 4.03
    d1 = 0.6
    s = 0.05
    rho = 1.225
    A = 0.503
    P_h = P0 * (1+3*v**2/U_tip**2)+Pi*math.sqrt(math.sqrt(1+v**4/(4*v0**4))-v**2/(2*v0**2))+0.5*d1*rho*s*A*v**3
    return P_h

def get_path_len(path):
    """ 计算路径的长度 """
    path_length = 0
    # print(path)
    for i in range(1, len(path)):
        node1_x = path[i][0]
        node1_y = path[i][1]
        node2_x = path[i - 1][0]
        node2_y = path[i - 1][1]
        path_length += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)
    return path_length



def get_dis(node1,node2):
    """ 计算路径的长度 """
    path_length = 0
    path_length += math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
    return path_length
path=[]
path_nomap=[]

