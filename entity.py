import numpy as np
import math


class User(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class UAV(object):
    def __init__(self, x, y, h):
        self.x = x
        self.y = y
        self.h = h

    def move_inside_test(self, phi, dist_half,dist_max):
        phi = phi
        dist = dist_half + dist_max/2
        self.x = self.x + dist * np.cos(phi)
        self.y = self.y + dist * np.sin(phi)
    # def move_inside_test(self, dx, dy, dist_max):
    #     move_dist = math.sqrt(dx ** 2 + dy ** 2)
    #     if move_dist > dist_max:
    #         dx *= dist_max / move_dist
    #         dy *= dist_max / move_dist

    #     new_x = self.x + dx
    #     new_y = self.y + dy

    #     # 强制边界控制
    #     new_x = max(0.0, min(new_x, 20))  # 假设地图尺寸是 10x10
    #     new_y = max(0.0, min(new_y, 20))

    #     self.x = new_x
    #     self.y = new_y



