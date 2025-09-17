from matplotlib import pyplot as plt
import numpy as np 
from Test import x0_user,y0_user
uav_num = 1
user_num = 6 
    
if __name__ == '__main__':
    np_ours_100 = np.load('results/trajectory/ours_100MB.npz')
    np_ours_200 = np.load('results/trajectory/ours_200MB.npz')
    np_ours_300 = np.load('results/trajectory/ours_300MB.npz')
    
    x_ours_100 = np.transpose(np_ours_100['x0_uav'])*100
    y_ours_100 = np.transpose(np_ours_100['y0_uav'])*100
    T_ours_100 = np.transpose(np_ours_100['T'])
    
    x_ours_200 = np.transpose(np_ours_200['x0_uav'])*100
    y_ours_200 = np.transpose(np_ours_200['y0_uav'])*100
    T_ours_200 = np.transpose(np_ours_200['T'])
    
    x_ours_300 = np.transpose(np_ours_300['x0_uav'])*100
    y_ours_300 = np.transpose(np_ours_300['y0_uav'])*100
    T_ours_300 = np.transpose(np_ours_300['T'])
    
    
    #画Radio Map 
    fig_1 = plt.figure(30)
    npzfile = np.load('results/datas/Radio_datas_A2G.npz')
    OutageMapActual = npzfile['arr_0']
    OutageMapActual_SINR = npzfile['arr_1']
    X_vec = npzfile['arr_2']  # [0,1....100]标号
    Y_vec = npzfile['arr_3']

    # plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual)
    # v = np.linspace(0, 1.0, 11, endpoint=True)
    plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, OutageMapActual_SINR)
    v = np.arange(-20, 36, 4)
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)
    
    for i in range(user_num): # 画用户位置
        plt.scatter(x0_user[i]*100, y0_user[i]*100, c='black', marker='^')
        
    ini_loc = [2.62, 2.65]
    end_loc = [18.95, 11.12]
    
    plt.scatter(ini_loc[0]*100, ini_loc[1]*100 , c='red', marker='o') #画起始点
    plt.scatter(end_loc[0]*100, end_loc[1]*100 , c='orange', marker='o') #画终点
    
    #绘制无人机轨迹
    plt.plot(x_ours_100[0][0:T_ours_100+1], y_ours_100[0][0:T_ours_100+1],c='black',label="Q=100MB")
    plt.plot(x_ours_200[0][0:T_ours_200+1], y_ours_200[0][0:T_ours_200+1],c='magenta',label="Q=200MB")
    plt.plot(x_ours_300[0][0:T_ours_300+1], y_ours_300[0][0:T_ours_300+1],c='blue',label="Q=300MB")
    plt.legend()
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    save_path = 'results/figs/Ours_Trajectory_A2G'
    plt.savefig(save_path)
    plt.close()
    