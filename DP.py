import math

import matplotlib.pyplot as plt
import numpy as np
import os
from entity import UAV, User
import radio_map as rad_env
from numpy import linalg as LA
## 动态规划法
class DP(object):
    
    def __init__(self, num_city, num_total, iteration, data, start_node=0, end_node=None):
        self.num_city = num_city
        self.location = data
        self.start_node = start_node
        self.end_node = end_node if end_node is not None else start_node
        self.dis_mat = self.compute_dis_mat(num_city, data)

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city)) # 距离矩阵
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)])) # compute the dis between location and target city
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算路径长度, goback:是否计算回到起始点的路径
    def compute_pathlen(self, path, dis_mat, goback=False):
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()

        if goback:
            result = dis_mat[a][b]
        else:
            result = 0.0
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 动态规划过程
    # def run(self):
    #     restnum = [x for x in range(1, self.num_city)] # [1..8]
    #     tmppath = [0]
    #     tmplen = 0
    #     while len(restnum) > 0:
    #         c = restnum[0]
    #         restnum = restnum[1:]
    #         if len(tmppath) <= 1:
    #             tmppath.append(c)
    #             tmplen = self.compute_pathlen(tmppath, self.dis_mat)
    #             continue

    #         insert = 0
    #         minlen = math.inf
    #         for i, num in enumerate(tmppath):
    #             a = tmppath[-1] if i == 0 else tmppath[i - 1]
    #             b = tmppath[i]
    #             tmp1 = self.dis_mat[c][a]
    #             tmp2 = self.dis_mat[c][b]
    #             curlen = tmplen + tmp1 + tmp2 - self.dis_mat[a][b]
    #             if curlen < minlen:
    #                 minlen = curlen
    #                 insert = i

    #         tmppath = tmppath[0:insert] + [c] + tmppath[insert:]
    #         tmplen = minlen # 轨迹长短
    #     return self.location[tmppath], tmplen
    def run(self):
        # 所有除起点和终点的中间节点
        middle_nodes = [x for x in range(self.num_city) if x != self.start_node and x != self.end_node]
        path = [self.start_node]  # 初始化路径，从起点开始

        while len(middle_nodes) > 0:
            best_insert_pos = 0
            best_insert_cost = math.inf
            node_to_insert = middle_nodes.pop(0)

            # 找出当前路径中最优的插入位置
            for i in range(1, len(path) + 1):  # 插入位置从1开始，确保start_node永远在首
                new_path = path[:i] + [node_to_insert] + path[i:]
                cost = self.compute_pathlen(new_path + [self.end_node], self.dis_mat, goback=False)
                if cost < best_insert_cost:
                    best_insert_cost = cost
                    best_insert_pos = i

            path = path[:best_insert_pos] + [node_to_insert] + path[best_insert_pos:]

        # 最后手动加入终点
        path.append(self.end_node)

        total_length = self.compute_pathlen(path, self.dis_mat, goback=False)
        return self.location[path], total_length



# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


def set_users():
    Users = []
    user_num = 9
    GT_loc = np.zeros([user_num, 3])
    users_path = 'Users_' + str(user_num) + '.txt'
    if os.path.exists(users_path):
        f = open(users_path, 'r') # 读取文件
        if f:
            user_loc = f.readline()
            user_loc = user_loc.split(' ')
            Users.append(User(float(user_loc[0]), float(user_loc[1]), float(user_loc[2])))
            GT_loc[len(Users) - 1] = np.array([float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
            while user_loc:
                user_loc = f.readline()
                if user_loc:
                    user_loc = user_loc.split(' ')
                    Users.append(User(float(user_loc[0]), float(user_loc[1]), float(user_loc[2])))
                    GT_loc[len(Users) - 1] = np.array([float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
            f.close()
    # else:
    #     f = open(users_path, 'w')
    #     creat = False
    #     while not creat:
    #         x = np.random.uniform(1, length - 1, 1)[0]
    #         y = np.random.uniform(1, width - 1, 1)[0]
    #         z = 0.0
    #         count = 0
    #         for index in range(urban_world.Build_num):
    #             x1 = HeightMapMatrix[index][0]
    #             x2 = HeightMapMatrix[index][1]
    #             y1 = HeightMapMatrix[index][2]
    #             y2 = HeightMapMatrix[index][3]
    #             if (x < x1 or x > x2) and (y < y1 or y > y2):
    #                 count += 1
    #                 continue
    #         if count == urban_world.Build_num:
    #             Users.append(User(x, y, z))
    #             GT_loc[len(Users) - 1] = np.array([x, y, z])
    #             f.writelines([str(x), ' ', str(y), ' ', str(z), '\n'])
    #         if len(Users) == user_num:
    #             creat = True
    #     f.close()
    return GT_loc[:,:-1]

def get_drection(Best_path,Best_path_next):
    if (Best_path[1]==Best_path_next[1])&(Best_path[0]==Best_path_next[0]):
        Phi=0
    else:
        Phi=np.arctan((Best_path_next[1]-Best_path[1])/(Best_path_next[0]-Best_path[0])) # to avoid dividing by 0
    Phi_deg=np.rad2deg(Phi)
    if (Best_path_next[1]>Best_path[1])&(Best_path_next[0]<Best_path[0]):
        Phi_deg=Phi_deg+180
    elif (Best_path_next[1]<Best_path[1])&(Best_path_next[0]<Best_path[0]):
        Phi_deg=Phi_deg-180
    return np.deg2rad(Phi_deg)

def step(ini_loc,drec,end_loc):
    distance = 0.20 # 一步距离
    DIST_TOLERANCE = 0.5 #
    next_loc = np.zeros(2)
    currect_loc = ini_loc
    out_time = 0.0
    while(True):
        next_loc[0] = currect_loc[0] + np.cos(drec) * distance
        next_loc[1] = currect_loc[1] + np.sin(drec) * distance
        Pout =get_empirical_outage(next_loc)
        out_time = out_time + Pout
        if LA.norm(next_loc-end_loc)<= DIST_TOLERANCE:
            print("################done###############")
            print("EndLoc：",end_loc)
            break
        else:
            currect_loc = next_loc
    return out_time

def get_empirical_outage(location):
    #location given in meters
    #convert the location to kilometer
    loc_km=np.zeros(shape=(1,3))
    loc_km[0,:2]=location/10
    loc_km[0,2]=0.1#UAV height in km95m
    Pout=rad_env.getPointMiniOutage(loc_km)
    return Pout[0]

#
if __name__ == "__main__":
    # Set parameters
    data = read_tsp('results/datas/Users_7.tsp')
    data = np.array(data)
    data = data[:, 1:]
    start_node = 0
    end_node = len(data) - 1
    model = DP(num_city=len(data), num_total=25, iteration=500, data=data.copy(), start_node=start_node, end_node=end_node)

    # model = DP(num_city=len(data), num_total=25, iteration=500, data=data.copy())
    Best_path, Best = model.run() # 返回位置顺序数组and轨迹长度
    print("Best_path",Best_path,Best)
    print("len(Best_path)",len(Best_path))
  
    print("最佳路径",Best_path)
    print('规划的路径长度:{}'.format(Best))
    comp_time = Best/0.20
   
    npzfile_sinr = np.load('results/datas/Radio_datas.npz')
    OutageMapActual = npzfile_sinr['arr_0'] 
    Y_vec2 = npzfile_sinr['arr_2']  # [0,1....100]标号
    X_vec2 = npzfile_sinr['arr_3']
    norm = plt.Normalize(vmin=-5, vmax=20)
    plt.contourf(np.array(Y_vec2) * 10, np.array(X_vec2) * 10, 1-OutageMapActual)
    v = np.linspace(-10, 30, 6, endpoint=True)
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
    plt.scatter(Best_path[0, 0]*1000, Best_path[0, 1]*1000, color='orange') # 画出起点
    plt.scatter(Best_path[-1, 0]*1000, Best_path[-1, 1]*1000, color='blue') # 画出终点
    plt.scatter(Best_path[1:-1, 0]*1000, Best_path[1:-1, 1]*1000, c='red', marker='^') # 画出任务点
    plt.plot(Best_path[:, 0]*1000, Best_path[:, 1]*1000,'b-')
    plt.xlabel('x (meter)', fontsize=14)    
    plt.ylabel('y (meter)', fontsize=14)
    plt.title('DP Path', fontsize=16)
    plt.savefig('results/figs/DP_path.png', dpi=300, bbox_inches='tight')
    plt.show()