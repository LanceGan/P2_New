from entity import UAV, User
import numpy as np
import os
import sys
import math
import copy
# import Radio_env_BS as rad_env
import radio_map as rad_env
from numpy import linalg as LA
from rural_world import Rural_world


class World(object):
    def __init__(self, length=10, width=10, uav_num=1, user_num=10,dist_max=0.10,
                 delta_t=0.5,t=200, uav_h=1,
                 data_size = 250,ini_loc=[10.93,4.4],end_loc=[15.5,18.8],
                 users_name='Users.txt'
                 ,BS_loc=[],
                 traverse_sequence=[]):
        self.length = length # 区域大小
        self.width = width
        self.uav_num = uav_num  # 总的无人机数目
        self.users_path = users_name # 用户位置存储路径
        self.user_num = user_num  # 用户数目
        self.Users = []
        self.UAVs = []
        self.T = t # 时间步
        self.t = 0
        self.max_x = length # 区域边界
        self.min_x = 0
        self.max_y = width
        self.min_y = 0
        self.uav_h = uav_h  # 无人机飞行高度
        self.dist_max = dist_max # max flying distance m
        self.delta_T = delta_t # 时隙长度
        self.fa = 0.0  # 出界
        self.r = 0.0  # 奖励
        self.terminal = False
        self.total_engy = 0.0
        self.out_time = 0.0
        self.NON_COVER_PENALTY = - 10
        self.Engy_w = 0.05

        self.initial_loc = ini_loc # 起点
        self.end_loc = end_loc # 终点
        self.distance = 0.5 # m
        self.SIR_THRESHOLD =3 # dB
        self.data_size_ini = data_size # 每个巡检点的数据量
        self.data_size = self.data_size_ini  # 每个巡检点的数据量
        self.data_rate = 0
        self.BandWidth  = 1
        self.transmit = False # 是否完成数据传输
        
        self.Traverse = traverse_sequence # 巡检点序列
        self.Traverse_order = 0 # 巡检点序列索引
        
        # =============Define the GT Distribution=======================
        
        self.GT_loc = BS_loc # 用户位置
        self.set_users()
        self.target = self.Users[self.Traverse[self.Traverse_order]-1]
        self.target_loc = np.array([self.target.x, self.target.y])  # 目标位置
        self.target_dis = LA.norm(self.initial_loc - self.target_loc) # 距离目标的距离
        self.target_pre_loc = np.array([self.initial_loc[0],self.initial_loc[1]]) #上一个目标的位置，初始为起点
        
        
        self.urban_world = Rural_world(self.GT_loc)
        self.HeightMapMatrix = self.urban_world.Buliding_construct()
        

    def set_users(self):
        self.Users =[]
        if os.path.exists(self.users_path): # 读写文件 # 读入用户位置
            f = open(self.users_path, 'r')
            if f:
                user_loc = f.readline()
                user_loc = user_loc.split(' ')
                self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                # self.GT_loc[len(self.Users) - 1] = np.array(
                #     [float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                while user_loc:
                    user_loc = f.readline()
                    if user_loc:
                        user_loc = user_loc.split(' ')
                        self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                        # self.GT_loc[len(self.Users) - 1] = np.array([float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                f.close()
        else:
           assert False, "Users file not found: " + self.users_path

    def reset_target(self):
        self.Traverse_order = 0
        self.target = self.Users[self.Traverse[self.Traverse_order]-1]
        self.target_loc = np.array([self.target.x, self.target.y])  # 目标位置
        self.target_dis = LA.norm(self.initial_loc - self.target_loc) # 距离目标的距离
        self.target_pre_loc = np.array([self.initial_loc[0],self.initial_loc[0]]) #上一个目标的位置，初始为起点
        
    # 重置位置，各种时间清0
    def reset(self):
        self.reset_target()
        self.set_uavs_loc()
        state = self.reset_state()
        self.data_size = self.data_size_ini
        self.transmit = False   
        self.t = 0
        self.fa = 0
        self.out_time = 0
        self.total_engy = 0.0
        return state,self.t

    def reset_loc(self):
        self.set_uavs_loc()
        state = self.reset_state()
        self.t = 0
        self.fa = 0
        self.out_time = 0
        self.total_engy = 0.0
        return state,self.t

    def set_uavs_test(self,location_x,location_y):
        self.UAVs = []
        for i in range(self.uav_num):
            x = location_x
            y = location_y
            h = self.uav_h
            self.UAVs.append(UAV(x, y, h))

    def reset_test(self, loc_x,loc_y):
        self.set_uavs_test(loc_x,loc_y) # 300m
        state = self.reset_state()
        self.data_size = self.data_size_ini
        self.t = 0
        self.fa = 0
        self.out_time = 0
        self.total_engy = 0.0
        return state,self.t


    def set_uavs_loc(self):
        self.UAVs = []
        for i in range(self.uav_num):
            x = self.initial_loc[0]
            y = self.initial_loc[1]
            h = self.uav_h
            self.UAVs.append(UAV(x, y, h))

    def set_uavs_Random(self):
        self.UAVs = []
        x_set = np.random.uniform(min(self.initial_loc[0], self.end_loc[0])-2,max(self.initial_loc[0], self.end_loc[0])+6,self.uav_num)
        y_set = np.random.uniform(min(self.initial_loc[1], self.end_loc[1])-2,max(self.initial_loc[1], self.end_loc[1])+6, self.uav_num)
        for i in range(self.uav_num):
            x = x_set[i]
            y = y_set[i]
            h = self.uav_h
            self.UAVs.append(UAV(x, y, h))
    # 重置环境
    
    def reset_state(self):
        s = np.zeros(self.uav_num*2 + 2+1+1+1+1)  
        # UAV location
        for i,uav in enumerate(self.UAVs):
            s[2*i] = uav.x                      
            s[2*i+1] = uav.y
        s[2] = self.target_loc[0]
        s[3] = self.target_loc[1]
        s[4] = self.target_dis
        s[5] = self.data_size
        s[6] = 1.0 if self.transmit else 0.0
        
        s[-1] = 0.0
        return s

    def update_state(self, engy,tar_loc,tar_dis):
        s_ = np.zeros(self.uav_num * 2 +2 +1 + 1+1+1) # 下一状态
        for i,uav in enumerate(self.UAVs):
            s_[i*2] = uav.x
            s_[i*2+1] = uav.y
            
        s_[2] = tar_loc[0]
        s_[3] = tar_loc[1]
        s_[4] = tar_dis
        s_[5] = self.data_size
        s_[6] = 1.0 if self.transmit else 0.0
        s_[-1] = engy
        return s_

    # 奖励表达式 平滑处理
    def get_reward(self,fa,fly_engy,dis_target,dis_pre_target,dis_forward,SINR):
        """
            fa: 出界惩罚
            fly_engy: 飞行能量
            dis_target: 当前位置与目标点距离
            dis_pre_target: 当前位置与上一个目标点距离
            dis_forward: 推进距离
            SINR: 信噪比
        """
               
        if dis_forward < 0:
            forward = -2
        else:
            forward = 1
       
        reward = forward*abs(dis_forward)*100 + 0.8*dis_pre_target - dis_target - fa
        # print("推进奖惩：",dis_forward*100,"距离目标奖励：",0.8*dis_pre_target - dis_target,"能量惩罚：",- self.Engy_w * fly_engy,"出界惩罚：",-fa)
        if SINR < self.SIR_THRESHOLD:
            # print("信噪比惩罚",self.NON_COVER_PENALTY)
            reward += self.NON_COVER_PENALTY
      
        return reward

    def get_empirical_outage(self, location,):
        loc_km=np.zeros(shape=(1,3))
        loc_km[0,:2]=location/10
        loc_km[0,2]=0.1
        Pout,Pout_SINR=rad_env.getPointMiniOutage(loc_km)
        # dateRate=rad_env.getPointDateRate(loc_km)
        return Pout[0],Pout_SINR[0]

    def get_date_rate(self, location):
        loc_km=np.zeros(shape=(1,3))
        loc_km[0,:2]=location/10
        loc_km[0,2]=0.1
        # Pout,Pout_SINRe=rad_env.getPointMiniOutage(loc_km)
        #Q print("获取速率的位置：",location)
        dateRate=rad_env.getPointDateRate(loc_km)
        return dateRate

    def step_inside(self, actions,s,t): # 输入动作，状态值，时间步,输出下一状态和其他标志
        fa = 0.0
        reward = 0.0
        fly_energy = 0.0
        self.t = t+1 # 时间步+1
        state_ = np.zeros(self.uav_num*2 +2+4)
        uav_location_pre = np.zeros([self.uav_num, 2])  # make a copy of the uav's location
        uav_location = np.zeros([self.uav_num, 2]) # uav current location
        for i, uav in enumerate(self.UAVs):
            uav_location_pre[i][0] = uav.x
            uav_location_pre[i][1] = uav.y
        if len(actions) == self.uav_num * 2:
            for i, uav in enumerate(self.UAVs):
                uav.move_inside_test(actions[0],actions[1],self.dist_max)  # execute the action执行飞行动作
            fly_energy = self.power((actions[1] + self.dist_max / 2) * 100 / self.delta_T) * self.delta_T
            # print("step_engy:",fly_energy)
            self.total_engy += fly_energy # compute energy
            for i, uav in enumerate(self.UAVs):
                penalty, bound = self.boundary_margin(uav) # 输出出界奖励,出界bound 为 false
                fa += penalty # 累计出界奖励Pob
                if not bound: # 如果出界
                    self.fa += 1
                    uav.x = uav_location_pre[i][0]      # the uav break the boundary constraint, the action is cancelled
                    uav.y = uav_location_pre[i][1]      # 无人机突破边界约束，动作取消
            
        
        for i, uav in enumerate(self.UAVs):
            uav_location[i][0] = uav.x
            uav_location[i][1] = uav.y
            # print("无人机当前位置: %f, %f",uav.x,uav.y)

        # Pout,Pout_SINR = self.get_empirical_outage(uav_location)  # 获取信号质量
        MaxSINR = self.get_date_rate(uav_location)  # 获取信号质量
        # 判断信噪比大小，避免出现负数
        
        if(MaxSINR >= self.SIR_THRESHOLD ):
            self.data_rate = self.BandWidth * np.log2(1 + 10**(MaxSINR/10.0))  # 数据传输速率
            if not self.transmit:
                self.data_size -= self.data_rate*self.delta_T
                reward += (self.data_rate*self.delta_T)
                if(self.data_size < 20):
                    self.transmit = True
                    print("=================================data transmit complete!!========================")
                    reward += 200
                    self.data_size = 0
      
        step_cur = LA.norm(uav_location - self.target_loc)  # 此位置与目标点距离
        self.target_dis = step_cur #更新目标距离
        
        step_pre = LA.norm(uav_location_pre - self.target_loc)  # 前一位置与目标点距离
        step_cur_pre = LA.norm(uav_location - self.target_pre_loc) # 此位置与前一目标点距离         
        reduce_dis = step_pre - step_cur #推进距离
        
        reward += self.get_reward(fa,fly_energy,step_cur,step_cur_pre,reduce_dis,MaxSINR) 
         
        
        if step_cur<=self.distance : # 达到检查点
            #只要到目标点了，就换下一个目标点，根据传输完成情况进行奖惩
            if self.transmit:  
                print("arrive and transmit complete")
                reward += 1000      
            else:
                print("arrive but not transmit complete, data size:",self.data_size)
                reward -= self.data_size*2   
                     
                
            self.target_pre_loc = self.target_loc #更新上一个目标点的位置        
            self.transmit = False
            self.data_size += self.data_size_ini # 重置数据量  
            
            if self.Traverse_order == len(self.Traverse): #到达终点了
                self.terminal = True
                self.done = True    
                print("=================================complete task!!========================")      
            else :                              
                self.terminal = False
                self.done = False               
                if self.Traverse_order < len(self.Traverse) - 1:  # 如果还有下一个目标点
                    print("=================================arrive point:",self.Traverse[self.Traverse_order])
                    self.Traverse_order += 1 #更新目标点
                    self.target = self.Users[self.Traverse[self.Traverse_order]-1]  # 更新目标点
                    self.target_loc = np.array([self.target.x, self.target.y])
                elif self.Traverse_order == len(self.Traverse) - 1:
                    print("=================================arrive point:",self.Traverse[self.Traverse_order])
                    self.Traverse_order += 1 #更新目标点
                    self.target_loc = self.end_loc
            # else :
                # reward -= 10 # 到达巡检点但数据未传输完成，给予惩罚
                # print("=================================arrive point but not transmit complete, data size:",self.data_size)            
        else:
            self.terminal = False
            
        done = False

        if self.terminal or self.t >= self.T: # 如果任务完成，或者飞行时间步已经满了
            done = True
            
        state_ = self.update_state(fly_energy,self.target_loc,self.target_dis)  # 进入下一状态
        self.r = reward
        # print("当前奖励:",reward)
        return state_, self.r,done,self.t,self.terminal

    #计算出界惩罚fa=1
    def boundary_margin(self, uav):
        bound = True
        v1 = 1.0
        alpha = 0.0
        beta = 100 / self.uav_num   #  Pob = 1/k惩罚项
        lx_plus = max(abs(uav.x - (self.max_x + self.min_x) / 2) - v1 * (self.max_x - self.min_x) / 2, 0.0)
        ly_plus = max(abs(uav.y - (self.max_y + self.min_y) / 2) - v1 * (self.max_y - self.min_y) / 2, 0.0)
        if lx_plus == 0.0 and ly_plus == 0.0:
            fa = 0.0
        else:
            fa = (alpha * (lx_plus ** 2 + ly_plus ** 2 ) + beta) * 1 # 出界惩罚
            bound = False # 出界标志
        return fa, bound

    # 推进能量
    def power(self,v):
        P0 = 79.8563
        Pi = 88.6279
        U_tip = 120
        v0 = 4.03
        d1 = 0.6
        s = 0.05
        rho = 1.225
        A = 0.503
        P_h = P0 * (1 + 3 * v ** 2 / U_tip ** 2) + Pi * math.sqrt(
            math.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2)) + 0.5 * d1 * rho * s * A * v ** 3
        return P_h