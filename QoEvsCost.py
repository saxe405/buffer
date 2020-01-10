import matplotlib.pyplot as plt
import re
import xlrd
__all__ = ["Identity", "Neighbours", "Probability_matrix"]
from Definitions import *
from transition_rules import Neighbours
from transition_rules import Probability_matrix
from transition_rules import cache_state_without_NP_and_CQ
from transition_rules import cache_state_NP
import seaborn as sns
import time
import numpy as np
from shutil import copyfile
from reward_rules import Networking_Cost_CQ
import os
from streaming import *
from head_movements import *
from common_analysis_fns import Action_table_builder

reward_penalty_coeffs = [ x for x in range(80,801,40) ]
reward_penalty_coeffs += [ x for x in range(900, 1501, 100)]
reward_penalty_coeffs = reward_penalty_coeffs[1:] #skip the first one as it has terrible performance
l_max_range = [0,5]
#user_gps = [0,1,2]
#video_gps = [0,1,2]
user_gps  = [1]
video_gps = [1]

Action_tables = {}

def save_figure(num, x_vecs, y_vecs, labels, x_lab, y_lab, file_name_suffix = ""):
    line_styles = ['-', ':', '-.']
    plt.style.use('seaborn')
    palette = plt.get_cmap('Set1')
    plt.figure(num)
    fig, ax = plt.subplots()
    for i in range(len(x_vecs)):
        if int(i/3) !=1:
            continue
        #ax.scatter( x_vecs[i], y_vecs[i] , color=palette(i%3), alpha=0.9,label = labels[i], marker =  ['+','*','.'][int(i/3)])        
        markr = ['d','*','.'][int(i/3)] + line_styles[int(i/3)]
        #ax.plot( x_vecs[i], y_vecs[i], markr , linewidth = 1, color=palette(i%3), label = labels[i], alpha=0.9 )
        ax.scatter( x_vecs[i], y_vecs[i] , color=palette(i%3), label = labels[i].split(', ')[1], alpha=0.9, marker =  ['d','*','.'][int(i%3)])
    
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.grid(True)
    filename = y_lab + "vs"+ x_lab + file_name_suffix
    filename = filename.replace(" ", "_")
    plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images_c/" + filename + ".png")

def save_sub_figure(num, x_vecs, y_vecs, labels, x_lab, y_lab, file_name_suffix = ""):
    plt.style.use('seaborn')
    palette = plt.get_cmap('Set1')
    plt.figure(num)
    fig, ax = plt.subplots()
    for i in range(len(x_vecs)):
        ax.scatter( x_vecs[i], y_vecs[i] , color=palette(i), label = labels[i], alpha=0.9, marker =  ['d','*','.'][i])
    
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.ylim((0.7,0.8))
    plt.xlim((2.6,3.7))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.grid(True)
    filename = y_lab + "vs"+ x_lab + file_name_suffix
    filename = filename.replace(" ", "_")
    plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/images_c/" + filename + ".png")
    
cachin_states = {}
streamin_states = {}
Cost_vec= {}
Freeze_prob = {}
FOV_quality = {}
state_vec = {}

Cost_vec_stream= {}
FOV_quality_stream = {}
Freeze_prob_stream = {}
state_vec_stream = {}
pi0 = 0.2
mtrust = 0.8
print("Building all the action tables")
for p in reward_penalty_coeffs:
    for l_m in l_max_range:
        for ug in user_gps:
            for vg in video_gps:
                Action_tables[(p,l_m,ug,vg)]   = Action_table_builder(Utility_book_name(p,l_m,ug,vg,0.2,0.8))
                cachin_states[(p,l_m,ug,vg)]  = starting_State_f(T = max_tiles(video_group = vg))                
                Cost_vec[(p,l_m,ug,vg)]       = [0]
                FOV_quality[(p,l_m,ug,vg)]    = []
                Freeze_prob[(p,l_m,ug,vg)]    = []
                state_vec[(p,l_m,ug,vg)]      = []

for ug in user_gps:
    for vg in video_gps:
        streamin_states[(ug,vg)]      = s_state(max_tiles(video_group = vg), Q, starting_State_f().CQ)
        Cost_vec_stream[(ug,vg)]       = [0]
        FOV_quality_stream[(ug,vg)]     = []
        Freeze_prob_stream[(ug,vg)]    = []
        state_vec_stream[(ug,vg)]      = []
                

print("Action tables ready")
t = 0

print("Starting with the initial phase")

while t < 10000:
    channel_quality = cachin_states[ (reward_penalty_coeffs[0], l_max_range[0], user_gps[0], video_gps[0]) ].CQ
    next_states = {}
    next_states_stream = {}
    for p in reward_penalty_coeffs:
        for l_m in l_max_range:
            for ug in user_gps:
                for vg in video_gps:
                    next_states[(p,l_m,ug,vg)] = cache_state_without_NP_and_CQ(Action_tables[(p,l_m,ug,vg)][cachin_states[(p,l_m,ug,vg)]], cachin_states[(p,l_m,ug,vg)], T = max_tiles(video_group = vg))
    for ug in user_gps:
        for vg in video_gps:
            next_states_stream[(ug,vg)] = stream_state_without_NP_and_CQ(streamin_states[(ug,vg)], video_group = vg)
    head_movement = {0 : np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[0].values())))], p=list(head_movement_lookup_table[0].values()))}
    head_movement[1] = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[1].values())))], p=list(head_movement_lookup_table[1].values()))
    head_movement[2] = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[2].values())))], p=list(head_movement_lookup_table[2].values()))
    if t%5 == 0:
        next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )
    
    for p in reward_penalty_coeffs:
        for l_m in l_max_range:
            for ug in user_gps:
                for vg in video_gps:
                    Cost_vec[(p,l_m,ug,vg)].append( Networking_Cost_CQ( Action_tables[(p,l_m,ug,vg)][cachin_states[(p,l_m,ug,vg)]], channel_quality))
                    #Network_util[(p,l_m,ug,vg)].append( Network_utilization(cachin_states[(p,l_m,ug,vg)], Action_tables[(p,l_m,ug,vg)][cachin_states[(p,l_m,ug,vg)]]))
                    #modify next states by applying the head movement transformations and channel quality variation
                    cachin_states[(p,l_m,ug,vg)] = cache_state_NP( next_states[(p,l_m,ug,vg)], head_movement[ug], next_channel_quality)
    for ug in user_gps:
        for vg in video_gps:
                    Cost_vec_stream[(ug,vg)].append( Networking_Cost_stream(streamin_states[(ug,vg)], next_states_stream[(ug,vg)]))
                    #Network_util_stream[(ug,vg)].append(Network_utilization_stream(streamin_states[(ug,vg)],next_states_stream[(ug,vg)]))
                    #modify next states by applying the head movement transformations and channel quality variation
                    streamin_states[(ug,vg)] = stream_state_NP( next_states_stream[(ug,vg)], head_movement[ug],next_channel_quality)
    t+=1
print("starting with the simulation phase")
t = 0
num_simulations = 50000
while t < num_simulations:
    channel_quality = cachin_states[ (reward_penalty_coeffs[0], l_max_range[0], user_gps[0], video_gps[0]) ].CQ    
    next_states = {}
    for p in reward_penalty_coeffs:
        for l_m in l_max_range:
            for ug in user_gps:
                for vg in video_gps:
                    next_states[(p,l_m,ug,vg)] = cache_state_without_NP_and_CQ(Action_tables[(p,l_m,ug,vg)][cachin_states[(p,l_m,ug,vg)]], cachin_states[(p,l_m,ug,vg)], T = max_tiles(video_group = vg))
    for ug in user_gps:
        for vg in video_gps:
            next_states_stream[(ug,vg)] = stream_state_without_NP_and_CQ(streamin_states[(ug,vg)], video_group = vg)
    
    head_movement = {0 : np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[0].values())))], p=list(head_movement_lookup_table[0].values()))}
    head_movement[1] = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[1].values())))], p=list(head_movement_lookup_table[1].values()))
    head_movement[2] = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table[2].values())))], p=list(head_movement_lookup_table[2].values()))
    
    if t%5 == 0:
        next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )

    for p in reward_penalty_coeffs:
        for l_m in l_max_range:
            for ug in user_gps:
                for vg in video_gps:
                    Cost_vec[(p,l_m,ug,vg)].append( Networking_Cost_CQ( Action_tables[(p,l_m,ug,vg)][cachin_states[(p,l_m,ug,vg)]], channel_quality))
                    #Network_util[(p,l_m,ug,vg)].append(  Network_utilization(cachin_states[(p,l_m,ug,vg)], Action_tables[(p,l_m,ug,vg)][cachin_states[(p,l_m,ug,vg)]]))
                    #FOV_hand[(p,l_m,ug,vg)].append(FOV_handover(cachin_states[(p,l_m,ug,vg)], video_group = vg))
                    Freeze_prob[(p,l_m,ug,vg)].append(Prob_screen_freeze(cachin_states[(p,l_m,ug,vg)], video_group = vg))
                    #FOV_vec[(p,l_m,ug,vg)].append(FOV_available(cachin_states[(p,l_m,ug,vg)], video_group = vg))
                    #modify next states by applying the head movement transformations and channel quality variation
                    cachin_states[(p,l_m,ug,vg)] = cache_state_NP( next_states[(p,l_m,ug,vg)], head_movement[ug], next_channel_quality)
                    if cachin_states[(p,l_m,ug,vg)].R == 1:
                        first_seg = cachin_states[(p,l_m,ug,vg)].n1  
                        next_action = Action_tables[(p,l_m,ug,vg)][cachin_states[(p,l_m,ug,vg)]]
                        if next_action[0] == 1:
                            first_seg +=next_action[1]
                        FOV_quality[(p,l_m,ug,vg)].append(FOV_quality_fn(first_seg, video_group = vg))
                    state_vec[(p,l_m,ug,vg)].append(cachin_states[(p,l_m,ug,vg)])
    for ug in user_gps:
        for vg in video_gps:
                    Cost_vec_stream[(ug,vg)].append( Networking_Cost_stream(streamin_states[(ug,vg)], next_states_stream[(ug,vg)]))
                    #modify next states by applying the head movement transformations and channel quality variation
                    streamin_states[(ug,vg)] = stream_state_NP( next_states_stream[(ug,vg)], head_movement[ug],next_channel_quality)
                    if streamin_states[(ug,vg)].R == 1:
                        first_seg = streamin_states[(ug,vg)].n1
                        first_seg +=stream_action(streamin_states[(ug,vg)], video_group = vg)
                        FOV_quality_stream[(ug,vg)].append(FOV_quality_stream_fn(first_seg, video_group = vg))
                    Freeze_prob_stream[(ug,vg)].append(Prob_screen_freeze_stream(streamin_states[(ug,vg)], video_group = vg))
                    state_vec_stream[(ug,vg)].append(streamin_states[(ug,vg)])
    t+=1
''''           
save_figure(1,  [ reward_penalty_coeffs]*9,
                [[ sum(Freeze_prob[(p,5,0,0)])/len(Freeze_prob[(p,5,0,0)]) for p in reward_penalty_coeffs], 
                 [ sum(Freeze_prob[(p,5,0,1)])/len(Freeze_prob[(p,5,0,1)]) for p in reward_penalty_coeffs], 
                 [ sum(Freeze_prob[(p,5,0,2)])/len(Freeze_prob[(p,5,0,2)]) for p in reward_penalty_coeffs], 
                 [ sum(Freeze_prob[(p,5,1,0)])/len(Freeze_prob[(p,5,1,0)]) for p in reward_penalty_coeffs], 
                 [ sum(Freeze_prob[(p,5,1,1)])/len(Freeze_prob[(p,5,1,1)]) for p in reward_penalty_coeffs], 
                 [ sum(Freeze_prob[(p,5,1,2)])/len(Freeze_prob[(p,5,1,2)]) for p in reward_penalty_coeffs], 
                 [ sum(Freeze_prob[(p,5,2,0)])/len(Freeze_prob[(p,5,2,0)]) for p in reward_penalty_coeffs], 
                 [ sum(Freeze_prob[(p,5,2,1)])/len(Freeze_prob[(p,5,2,1)]) for p in reward_penalty_coeffs], 
                 [ sum(Freeze_prob[(p,5,2,2)])/len(Freeze_prob[(p,5,2,2)]) for p in reward_penalty_coeffs], 
                ],
                ["ug 0, vg 0", "ug 0, vg 1", "ug 0, vg 2", "ug 1, vg 0","ug 1, vg 1","ug 1, vg 2","ug 2, vg 0","ug 2, vg 1","ug 2, vg 2"],
                "penalty parameter", "Freeze Probability")

fn = 0
ind1 = 0
ind2 = 0
var1 = [Cost_vec][ind1]
var2 = [FOV_quality ][ind2]        
fn+=1
save_figure(fn,
      [[ sum(var1[(p,5,0,0)])/len(var1[(p,5,0,0)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,5,0,1)])/len(var1[(p,5,0,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,5,0,2)])/len(var1[(p,5,0,2)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,5,1,0)])/len(var1[(p,5,1,0)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,5,1,1)])/len(var1[(p,5,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,5,1,2)])/len(var1[(p,5,1,2)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,5,2,0)])/len(var1[(p,5,2,0)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,5,2,1)])/len(var1[(p,5,2,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,5,2,2)])/len(var1[(p,5,2,2)]) for p in reward_penalty_coeffs], 
      ], 
      [ [ sum(var2[(p,5,0,0)])/len(var2[(p,5,0,0)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,5,0,1)])/len(var2[(p,5,0,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,5,0,2)])/len(var2[(p,5,0,2)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,5,1,0)])/len(var2[(p,5,1,0)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,5,1,1)])/len(var2[(p,5,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,5,1,2)])/len(var2[(p,5,1,2)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,5,2,0)])/len(var2[(p,5,2,0)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,5,2,1)])/len(var2[(p,5,2,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,5,2,2)])/len(var2[(p,5,2,2)]) for p in reward_penalty_coeffs], 
      ],
      ["ug 0, vg 0", "ug 0, vg 1", "ug 0, vg 2", "ug 1, vg 0","ug 1, vg 1","ug 1, vg 2","ug 2, vg 0","ug 2, vg 1","ug 2, vg 2"],
      ["Networking Cost", "Network Utilization"][ind1], ["Freeze Probability", "FOV len available"][ind2])


#for ind1 in range(2):
#    for ind2 in range(2):
var1 = [Cost_vec][ind1]
var1p = [Cost_vec_stream][ind1]
var2 = [FOV_quality][ind2]
var2p = [FOV_quality_stream][ind2]
fn+=1
save_figure(fn,
      [[ sum(var1[(p,5,1,0)])/len(var1[(p,5,1,0)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,0,1,0)])/len(var1[(p,0,1,0)]) for p in reward_penalty_coeffs], 
       [ sum(var1p[(1,0)])/len(var1p[(1,0)]) ]*len(reward_penalty_coeffs), 
       [ sum(var1[(p,5,1,1)])/len(var1[(p,5,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,0,1,1)])/len(var1[(p,0,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1p[(1,1)])/len(var1p[(1,1)]) ]*len(reward_penalty_coeffs), 
       [ sum(var1[(p,5,1,2)])/len(var1[(p,5,1,2)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,0,1,2)])/len(var1[(p,0,1,2)]) for p in reward_penalty_coeffs], 
       [ sum(var1p[(1,2)])/len(var1p[(1,2)]) ]*len(reward_penalty_coeffs), 
      ], 
      [ [ sum(var2[(p,5,1,0)])/len(var2[(p,5,1,0)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,0,1,0)])/len(var2[(p,0,1,0)]) for p in reward_penalty_coeffs], 
        [ sum(var2p[(1,0)])/len(var2p[(1,0)]) ]* len(reward_penalty_coeffs), 
        [ sum(var2[(p,5,1,1)])/len(var2[(p,5,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,0,1,1)])/len(var2[(p,0,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2p[(1,1)])/len(var2p[(1,1)]) ]* len(reward_penalty_coeffs), 
        [ sum(var2[(p,5,1,2)])/len(var2[(p,5,1,2)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,0,1,2)])/len(var2[(p,0,1,2)]) for p in reward_penalty_coeffs], 
        [ sum(var2p[(1,2)])/len(var2p[(1,2)]) ] *len(reward_penalty_coeffs), 
      ],
      ["vg 0, predictive-caching", "vg 0, short-term caching", "vg 0, no-caching", "vg 1, predictive-caching","vg 1, short-term caching","vg 1, no-caching","vg 2, predictive-caching","vg 2, short-term caching","vg 2, no-caching"],
       ["Networking Cost", "Network Utilization"][ind1], ["Freeze Probability", "FOV len available"][ind2], file_name_suffix = "policies_VG")

#for ind1 in range(2):
#    for ind2 in range(2):
var1 = [Cost_vec][ind1]
var1p = [Cost_vec_stream][ind1]
var2 = [FOV_quality][ind2]
var2p = [FOV_quality_stream][ind2]
fn+=1
save_figure(fn,[[ sum(var1[(p,5,0,1)])/len(var1[(p,5,0,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,0,0,1)])/len(var1[(p,0,0,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1p[(0,1)])/len(var1p[(0,1)]) ]*len(reward_penalty_coeffs), 
       [ sum(var1[(p,5,1,1)])/len(var1[(p,5,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,0,1,1)])/len(var1[(p,0,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1p[(1,1)])/len(var1p[(1,1)]) ]*len(reward_penalty_coeffs), 
       [ sum(var1[(p,5,2,1)])/len(var1[(p,5,2,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,0,2,1)])/len(var1[(p,0,2,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1p[(2,1)])/len(var1p[(2,1)]) ]*len(reward_penalty_coeffs), 
      ], 
      [ [ sum(var2[(p,5,0,1)])/len(var2[(p,5,0,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,0,0,1)])/len(var2[(p,0,0,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2p[(0,1)])/len(var2p[(0,1)]) ]* len(reward_penalty_coeffs), 
        [ sum(var2[(p,5,1,1)])/len(var2[(p,5,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,0,1,1)])/len(var2[(p,0,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2p[(1,1)])/len(var2p[(1,1)]) ]* len(reward_penalty_coeffs), 
        [ sum(var2[(p,5,2,1)])/len(var2[(p,5,2,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,0,2,1)])/len(var2[(p,0,2,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2p[(2,1)])/len(var2p[(2,1)]) ] *len(reward_penalty_coeffs), 
      ],
      ["ug 0, predictive-caching", "ug 0, short-term caching", "ug 0, no-caching", "ug 1, predictive-caching","ug 1, short-term caching","ug 1, no-caching","ug 2, predictive-caching","ug 2, short-term caching","ug 2, no-caching"],
       ["Networking Cost", "Network Utilization"][ind1], ["Freeze Probability", "FOV len available"][ind2], file_name_suffix = "policies_UG")

save_sub_figure(1,  [ reward_penalty_coeffs],
                [[ sum(Freeze_prob[(p,5,1,1)])/len(Freeze_prob[(p,5,1,1)]) for p in reward_penalty_coeffs], 
                ],
                [""],
                "penalty parameter", "FOV quality")
'''
fn = 0
var1 = Cost_vec
var1p = Cost_vec_stream
var2 = FOV_quality
var2p = FOV_quality_stream
fn+=1
save_sub_figure(fn,
      [
       [ sum(var1[(p,5,1,1)])/len(var1[(p,5,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1[(p,0,1,1)])/len(var1[(p,0,1,1)]) for p in reward_penalty_coeffs], 
       [ sum(var1p[(1,1)])/len(var1p[(1,1)]) ]*len(reward_penalty_coeffs), 
      ], 
      [
        [ sum(var2[(p,5,1,1)])/len(var2[(p,5,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2[(p,0,1,1)])/len(var2[(p,0,1,1)]) for p in reward_penalty_coeffs], 
        [ sum(var2p[(1,1)])/len(var2p[(1,1)]) ]* len(reward_penalty_coeffs), 
      ],
      ["predictive-buffering", "short-term buffering", "no-buffering"],
       "Resource usage", "FOV quality", file_name_suffix = "policies")
