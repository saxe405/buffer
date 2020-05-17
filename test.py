# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:55:45 2019

@author: sapoorv

"""

from Definitions import *
from channel import *
from matplotlib import pyplot as plt

def F_new(x,block_num = 1, video_group = args.video_group):    
    val = x/12

    return(min(val,1))

Cost_CQ_new = [120, 10, 0.5]
x_vals = [ x for x in range(1,max_tiles()+1)]

Reward_matrix_coeff = [2000,440,90]
seg1 = []
seg2 = []
seg3 = []
for til in x_vals:
    seg1.append((F_new(til,1)- F_new(til-1,1))*Reward_matrix_coeff[0])
    seg2.append((F_new(til,2)- F_new(til-1,2))*Reward_matrix_coeff[1])
    seg3.append((F_new(til,3)- F_new(til-1,3))*Reward_matrix_coeff[2])


plt.figure(1)
plt.plot(x_vals,seg1, c='blue', linewidth=2)
plt.plot(x_vals,seg2, c='green', linewidth=2)
plt.plot(x_vals,seg3, c='black', linewidth=2)
ytics = []

#boundary of segment1
plt.axhline(seg1[-1]/4, c = 'red', linestyle = '--')

#boundary segment 2
plt.axhline(seg2[-1]/4, c = 'red', linestyle = '--')
#boundary segment 3
plt.axhline(seg3[-1]/4, c = 'red', linestyle = '--')

#netwoking cost
plt.axhline(Cost_CQ_new[1], c = 'magenta')
plt.axhline(Cost_CQ_new[2], c = 'magenta')
ytics += Cost_CQ_new[1:]

plt.xlabel("Tile number")
plt.ylabel("delta reward")
plt.title("Comparing change in rewards and cost per tile")

plt.yticks(ytics)
plt.show()
print(ytics)
    