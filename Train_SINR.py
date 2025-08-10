import torch
from TD3 import TD3
import ReplayBuffer
from World import World
import numpy as np
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
import argparse

def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_h", help="set the UAV height", type=float,default=1.0) # 无人机高度
    parser.add_argument("--gamma", help="set the gamma", type=float,default=0.99)
    parser.add_argument("--buffer", help="set the buffer", type=int,default=200000)
    parser.add_argument("--net_width", help="set the net width",type=int,default=256)

    return parser


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

parser = create_parser()
args = parser.parse_args()
gamma = args.gamma
net_width = args.net_width
buffer = args.buffer
uav_h = args.uav_h
train_path = 'logs/exp/'
mkdir(train_path)
writer = SummaryWriter(train_path)
result_path = 'Result/'
mkdir(result_path[:-1])
# train
np.random.seed(1)
torch.manual_seed(1)

uav_num = 1
user_num = 6
T = 1000 #飞行步数
Length = 20  #区域范围2km,  100 m per 1
Width = 20
s_dim = uav_num*2 +2 +1 +1   # [uav.x,uav.y,target.x,target.y,dis_target,engy]
a_dim = uav_num*2  # [theta,theta_z, upsilon]

# In this paper, we consider the flight time slot in e2ach step is a constant.
# In this case, the flight speed is completely equivalent to the flight distance variable.
# Thus, in the simulation, we use the flight distance as the action.
# You also can use the flight speed as the action directly.
V_max = 0.50 #最大飞行速度 V_max
delta_t = 0.5 # 离散时隙长度s
dist_max = delta_t * V_max # 10m

max_action = np.array([math.pi,dist_max/2])  # {(-pi,pi); (-pi/2,pi/2);(-25m, 25m)}
total_episode = 8000
sample_episode = 1000
memory_size = T * sample_episode  # The capacity of experience replay buffer
train_memory_size = T * 10  # The capacity of experience replay buffer


data_size =600 # 每个巡检点的数据量

"""
 [ 8.22  4.34]
 [ 6.5  11.43]
 [ 4.3  17.9 ]
 [15.9  17.1 ]
 [11.34 12.8 ]
 [14.52  7.05]
 [18.7  11.7 ]]
"""
ini_loc = [2.62 ,2.65 ]
end_loc = [18.95 ,11.12]
BS_loc = np.array([[4.77,4.56,0.25],[14.89,4.89,0.25],[4.63,13.78,0.25],[13.48,14.89,0.25]])
trave_order = [1,6,2,3,5,4]
world = World(length=Length, width=Width, uav_num=uav_num,dist_max=dist_max,delta_t=delta_t,t=T,uav_h=uav_h,
              data_size=data_size,ini_loc=ini_loc,end_loc=end_loc,
              users_name='results/datas/Users_'+str(user_num)+'.txt',
              BS_loc=BS_loc,traverse_sequence=trave_order)


expl_noise = 0.6  # random noise std
t1 = time.time()

env_with_Dead = True
kwargs = {
    "env_with_Dead": env_with_Dead,
    "state_dim": s_dim,
    "action_dim": a_dim,
    "max_action": max_action,
    "train_path":train_path,
    "gamma": gamma,
    "net_width": net_width,
    "a_lr": 1e-4,
    "c_lr": 1e-3,
    "Q_batchsize": 256,
}
model = TD3(**kwargs)
model_path = 'Model/'
mkdir(model_path[:-1])
replay_buffer = ReplayBuffer.ReplayBuffer(s_dim, a_dim, max_size=int(buffer))

x = y = np.arange(0,Width,0.01)
x, y = np.meshgrid(x,y)


def draw_location(x_uav, y_uav,t, savepath):
    x_uav = np.transpose(x_uav)
    y_uav = np.transpose(y_uav)
    plt.figure(facecolor='w', figsize=(20, 20))
    # for i in range(user_num):
    #     if s[user_num+i]==1.0:
    #         plt.scatter(x_user[i], y_user[i], c='red', marker='x', s=150, linewidths=4)
    #     else:
    #         plt.scatter(x_user[i], y_user[i], c='black', marker='x', s=150, linewidths=4)
    for i in range(uav_num):
        plt.plot(x_uav[i][0:t+1], y_uav[i][0:t+1], c='blue', marker='.', linewidth=3.5, markersize=7.5)
        plt.plot(x_uav[i][t], y_uav[i][t], c='green', marker='.', markersize=12.5)
    for index in range(world.urban_world.Build_num):
        x1 = world.HeightMapMatrix[index][0]
        x2 = world.HeightMapMatrix[index][1]
        y1 = world.HeightMapMatrix[index][2]
        y2 = world.HeightMapMatrix[index][3]
        XList = [x1, x2, x2, x1, x1]
        YList = [y1, y1, y2, y2, y1]
        plt.plot(XList, YList, 'r-')
    plt.title('location', fontsize=30)
    plt.xlim((0, Length))
    plt.ylim((0, Width))
    plt.grid()
    plt.savefig(savepath)
    plt.close()


x0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
y0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
z0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
location = np.zeros([3, world.uav_num])

ep_rewards=[]
ep_avg_v=[]
for episode in tqdm(range(1, total_episode+1),ascii=True, unit='episodes'):
    t3 = time.time()
    s, t = world.reset()
    fly_dis = 0.0
    for i, uav in enumerate(world.UAVs):
        location[0][i] = uav.x
        location[1][i] = uav.y
        location[2][i] = uav.h
    x0_uav[episode][0] = location[0]
    y0_uav[episode][0] = location[1]
    z0_uav[episode][0] = location[2]
    sum_reward = 0
    sum_fly_v = 0
    actions = np.zeros(2 * uav_num) # 动作空间自由度
    done = False
    expl_noise *= 0.999
    while not done:
        a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=a_dim)
             ).clip(-max_action, max_action)  # obtain a new action
        # a = model.select_action(s)
        fly_dis += (a[1] + dist_max / 2)
        sum_fly_v  += (a[1] + dist_max / 2)*100 / delta_t # 速度
        #print("",)
        s_, r, done,t,terminal = world.step_inside(a,s,t)
        # s = s_  # update state
        sum_reward += r
        replay_buffer.add(s, a, r, s_, terminal)  # put a transition in buffer

        if replay_buffer.size > train_memory_size: model.train(replay_buffer)

        s = s_  # update state
        
        for i, uav in enumerate(world.UAVs):
            location[0][i] = uav.x
            location[1][i] = uav.y
            location[2][i] = uav.h
        x0_uav[episode][t] = location[0]
        y0_uav[episode][t] = location[1]
        z0_uav[episode][t] = location[2]
        # print(uav.x, uav.y, uav.h)
        
        # if done and agent.pointer <= memory_size: # and episode % 5 == 0:
        #     print('Episode:', episode, 'Epoch:', t, 'total_Reward: %.4f' % sum_reward, 'reward: ', r,'fa',world.fa,'raw_r',world.r,'num_fa2',num_fa2, 'Explore:%.2f'% epsino)
        #     print('sum_cover:',world.sum_cover)
        #if done and agent.pointer > memory_size: #and episode % 5 == 0:
        # print("Tatget point: %d",world.Traverse[world.Traverse_order])
        
    print('Episode:', episode, 'Epoch:', t,'total_Reward: %.4f' % sum_reward, 'Explore:%.2f'% expl_noise)
    print("iniloc", world.initial_loc,world.end_loc,world.data_size,"Distance:",fly_dis,"Engy",world.total_engy,"avg_v",sum_fly_v/t,"Fa",world.fa)
    # print('loss_a:', agent.loss_a, 'td_error:', agent.td_error)
    # print('sum_cover:',world.sum_cover)
    writer.add_scalar('total_reward', sum_reward, episode)
    ep_rewards.append(sum_reward)  # 保存每集的奖励
  #  if agent.pointer > memory_size:
    #draw_location(x0_uav[episode], y0_uav[episode],t, x0_user, y0_user,
    #             result_path + 'UAVPath_Users_%s.png' % str(episode).zfill(2),s)
    if episode % 50 == 0 : # and episode >= 2000:
        model.save(episode,model_path)
    if episode %1000 == 0 and episode >0:
        expl_noise = 0.6
       # torch.save(agent.Critic_eval.state_dict(), 'Critic_model' + str(episode) + '_' + str(user_num) + '.pkl')
    # print(time.time()-t3)
#print('Running time:', time.time() - t1)
#######画奖励收敛图
def get_moving_average(mylist, N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_aves.append(moving_ave)
    return moving_aves



fig=plt.figure()
plt.xlabel('Episode')
plt.ylabel('Return per episode')
plt.plot(range(len(ep_rewards)), ep_rewards)
N = 200

return_mov_avg = get_moving_average(ep_rewards, N)
plt.plot(np.arange(len(return_mov_avg)) + N, return_mov_avg, 'r-', linewidth=5)
fig.savefig('reward.jpg')
# plt.show()

np.savez('UAV_TD3',return_mov_avg,ep_rewards)