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

penalty_param = 160
l_max = 5
ug = 1
vg = 1

book = Utility_book_name(penalty_param,l_max, ug,vg,0.2,0.8)
Action_table = Action_table_builder(book)
plt.figure(1)
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
    #x.append(x_val)
    size_vec.append((action[1]+1)**2)
    y_val = state.CQ
    
    val = 20*(action[1]+1)**2
    if (x_val,y_val) in all_actions:
        val = max( val, all_actions[(x_val,y_val)] )
    if action[0] == 0:
        val = 1
    all_actions[(x_val,y_val)] = val #number of tiles downloaded
    colors[(x_val,y_val)] = action[0]
    if x_val > 15 and action[0] == 1:
        print(state, action, Identity(state.n1, video_group = vg))   
    # Change color with c and alpha. I map the color to the X axis value.
x = [a[0] for a in all_actions.keys()]
y = [a[1] for a in all_actions.keys()]

plt.scatter(x,y,s=list(all_actions.values()), c=list(colors.values()), cmap=plt.cm.get_cmap('Paired', 3), alpha=0.4, edgecolors="black", linewidth=2)
plt.colorbar(ticks = range(3), label = 'download segment')
# Add titles (main and on axis)
plt.xlabel("buffer_state")
plt.ylabel("Channel quality")
plt.title("Decision Visual")
plt.axvspan(0, max_tiles(video_group = vg) , alpha=0.5, color='seashell')
plt.axvspan(max_tiles(video_group = vg)+1,max_tiles(video_group = vg)*2, alpha=0.5, color='aliceblue')
plt.text(3, -0.4, 'segment 1',
        verticalalignment='bottom', horizontalalignment='left',        
        color='coral', fontsize=15)
plt.text(17, -0.4, 'segment 2',
        verticalalignment='bottom', horizontalalignment='left',        
        color='dodgerblue', fontsize=15)
plt.yticks([0,1,2])
plt.show()
plt.tight_layout()
plt.grid(True)
file_name = "Decision_visual"
plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images/" + file_name +".png")