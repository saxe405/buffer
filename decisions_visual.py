# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:36:20 2019

@author: sapoorv
"""
from Definitions import Identity, Utility_book_name
from common_analysis_fns import Action_table_builder
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

penalty_param = 120
l_max = 5
ug = 1
vg = 1

book = Utility_book_name(penalty_param,l_max, ug,vg,0.2,0.8,1,'W3_all')
Action_table = Action_table_builder(book)

x = []
y = []
size_vec = []
colors = {}
all_actions = {}

for state in Action_table.keys():
    action = Action_table[state]
    x_val = state.n1
    if x_val >= max_tiles(video_group = vg):
        x_val = x_val + state.n2
        if Identity(state.n2, block_num = 2, video_group = vg) == 1:
            x_val+=1
    
    size_vec.append((action[1]+1)**2)
    y_val = state.CQ
    
    val = 8*(action[1]+1)**2
    
    if action[0] == 0:
        val = 1

    if (x_val,y_val) not in all_actions:
        all_actions[(x_val,y_val)] = val
        colors[(x_val,y_val)] = action[0]
    else:
        if all_actions[(x_val,y_val)] < val:
            all_actions[(x_val,y_val)] = val
            colors[(x_val,y_val)] = action[0]
    
    if x_val > 15 and action[0] == 1:
        print(state, action, Identity(state.n1, video_group = vg))   
    # Change color with c and alpha. I map the color to the X axis value.
x = [a[0] for a in all_actions.keys()]
y = [a[1] for a in all_actions.keys()]

plt.figure(1)

plt.scatter(x,y,s=list(all_actions.values()), c=list(colors.values()), cmap=plt.cm.get_cmap('RdBu',4), alpha=0.4, edgecolors="black", linewidth=2)
plt.axvline(max_tiles(video_group = vg)+0.5)
plt.axvline(max_tiles(video_group = vg)*2+0.5)
plt.colorbar(ticks = range(4), label = 'download segment')
# Add titles (main and on axis)
plt.xlabel("buffer_state")
plt.ylabel("Channel quality")
plt.title("Decision Visual")

plt.text(3, 0.4, 'buffer seg 1',
        verticalalignment='bottom', horizontalalignment='left',        
        color='coral', fontsize=10)
plt.text(15, 0.4, 'buffer seg 2',
        verticalalignment='bottom', horizontalalignment='left',        
        color='dodgerblue', fontsize=10)

plt.text(25, 0.1, 'long term segment',
        verticalalignment='bottom', horizontalalignment='left',        
        color='dodgerblue', fontsize=7, rotation = 90)
plt.yticks([0,1,2])
plt.xticks([0,6,12,13,18,24,25],[0,6,12,1,6,12,1])
plt.show()
plt.tight_layout()
plt.grid(False)
file_name = "Decision_visual"
plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images_c/" + file_name +".pdf")