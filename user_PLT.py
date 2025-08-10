import os
import numpy as np
import matplotlib.pyplot as plt
user_num=8
from matplotlib.patches import Ellipse, Circle
users_path = r'/home/lancegan/Datas/Codes/Python/P2_New/results/datas/Users_7.txt'


x_user = []
y_user = []
f = open(users_path, 'r')


if f:
    for j in range(user_num):
        user_loc = f.readline()
        # print("user_loc", user_loc)
        user_loc = user_loc.split(' ')
        x_user.append(float(user_loc[0]))
        # print("x_user",x_user)
        y_user.append(float(user_loc[1]))
    fig, ax = plt.subplots()
    npzfile = np.load(r'results/datas/Radio_datas.npz')
    print(npzfile.files)
    OutageMapActual = npzfile['arr_0']
    X_vec = npzfile['arr_2']  # [0,1....100]标号
    Y_vec = npzfile['arr_3']

    plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual)
    v = np.linspace(0, 1.0, 11, endpoint=True)
    cbar = plt.colorbar(ticks=v)
    cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)

    cir = np.zeros(shape=(user_num, 1))
    for i in range(user_num):  # 画用户位置
        if i !=0 and i != user_num-1:
            
            plt.scatter(x_user[i]*1000, y_user[i]*1000 , c='red', marker='^')
        elif i==0:
            plt.scatter(x_user[i]*1000, y_user[i]*1000 , c='orange')
        elif i==user_num-1:
            plt.scatter(x_user[i]*1000, y_user[i]*1000 , c='blue')
        #  cir = Circle(xy=(x_user[i]*1000, y_user[i]*1000), radius=60, alpha=1,fill=False)
        #  ax.add_patch(cir)

    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.savefig('results/figs/Radio_map_user.png', dpi=300, bbox_inches='tight')
    # plt.axis('equal')
    plt.show()




