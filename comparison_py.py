import matplotlib.pyplot as plt
import re
import xlrd
import plotly as py
import plotly.graph_objs as go
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

def read_traces(trace_num):
	f=open("5G_traces/WiGig_"+ str(trace_num) + ".txt", "r")
	f.contents().split('\n')

def save_plot(plot_num, Performance_index, y_lab, x_lab = 'Time'):
	plt.figure(plot_num)
	fig, ax = plt.subplots()

	ax.plot([0.1*x for x in range(len(Performance_index))], [x[0] for x in Performance_index], 'b-', label = 'predictive caching' )
	ax.plot([0.1*x for x in range(len(Performance_index))], [x[1] for x in Performance_index], 'b-', label = 'streaming' )
	
	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
	legend.get_frame().set_facecolor('#00FFCC')
	plt.tight_layout()
	plt.grid(True)
	plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/"+ y_lab +".png")

data = xlrd.open_workbook('Updated Utilities_1.0.xlsx')
table = data.sheets()[0]
ncols = table.ncols
num_iterations = int((ncols-8)/2)

Action_table = {}

for i in range(table.nrows):
	vals = table.row_values(i)		
	s = new_state(int(vals[0]),int(vals[1]),int(vals[2]),int(vals[3]),int(vals[4]),int(vals[5]))
	a = (int(vals[ncols-1][0]), int(vals[ncols-1][2:]) )	
	Action_table[s] = a
print("Action table ready")

t = 0
cachin_state = starting_State
streamin_state = s_state(T,Q,cachin_state.CQ)
Cost_vec = [[0,0]]
QoE_vec = [[0,0]]
Network_util = [[0,0]]
print("Starting with the initial phase")
#print(list(head_movement_lookup_table.values()))
while t < 1000:
	t+=1
	best_c_action = Action_table[cachin_state]
	channel_quality = cachin_state.CQ
	next_c_state = cache_state_without_NP_and_CQ(best_c_action,cachin_state)	
	next_s_state = stream_state_without_NP_and_CQ(streamin_state)

	head_movement = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table.values())))], p=list(head_movement_lookup_table.values()))	
#	a[np.random.choice([x for x in range(1,len(list(a.values()))+1)], p = list(a.values()))]
	#print(head_movement)
	next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )
	#print(next_channel_quality)

	Cost_vec.append([	Cost_vec[-1][0]*gamma + Networking_Cost_CQ(best_c_action,streamin_state.CQ),
						Cost_vec[-1][1]*gamma + Networking_Cost_stream(streamin_state, next_s_state) 
		])
	Network_util.append([	gamma*Network_util[-1][0]+ Network_utilization(cachin_state,best_c_action), 
							gamma*Network_util[-1][1]+ Network_utilization_stream(streamin_state,next_s_state),
		])
	
	#modify next states by applying the head movement transformations and channel quality variation
	cachin_state = cache_state_NP( next_c_state,head_movement,next_channel_quality)
	streamin_State = stream_state_NP(next_s_state,head_movement,next_channel_quality)

state_vec = [[cachin_state, streamin_state]]
FOV_vec = [[FOV_available(cachin_state),FOV_available_stream(streamin_state)]]
Cost_vec = [Cost_vec[-1]]
QoE_vec = [QoE_vec[-1]]
Network_util = [Network_util[-1]]
FOV_hand = [[ FOV_handover(cachin_state), FOV_handover_stream(streamin_state)]]
Freeze_prob = [[Prob_screen_freeze(cachin_state),Prob_screen_freeze_stream(streamin_state)]]
print("starting with the simulation phase")
t = 0
while t < 200:
	t+=1

	best_c_action = Action_table[cachin_state]
	channel_quality = cachin_state.CQ
	next_c_state = cache_state_without_NP_and_CQ(best_c_action,cachin_state)	
	next_s_state = stream_state_without_NP_and_CQ(streamin_state)

	head_movement = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table.values())))], p=list(head_movement_lookup_table.values()))	
	next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )
	
	print(head_movement)
	print(next_channel_quality)

	Cost_vec.append([	Cost_vec[-1][0]*gamma + Networking_Cost_CQ(best_c_action,streamin_state.CQ),
						Cost_vec[-1][1]*gamma + Networking_Cost_stream(streamin_state, next_s_state) 
					])
	Network_util.append([	gamma*Network_util[-1][0]+ Network_utilization(cachin_state,best_c_action), 
							gamma*Network_util[-1][1]+ Network_utilization_stream(streamin_state,next_s_state),
		])
	
	FOV_hand.append( [ FOV_handover(next_c_state), FOV_handover_stream(next_s_state)])
	Freeze_prob.append([Prob_screen_freeze(next_c_state),Prob_screen_freeze_stream(next_s_state)])

	
	FOV_vec.append([FOV_available(next_c_state),FOV_available_stream(next_s_state)])
	
	#modify next states by applying the head movement transformations and channel quality variation
	cachin_state 	= cache_state_NP( next_c_state,head_movement,next_channel_quality)
	streamin_State = stream_state_NP(next_s_state,head_movement,next_channel_quality)
	state_vec.append([cachin_state, streamin_state])

save_plot(1,FOV_vec, "FOV_available")
save_plot(2,Cost_vec, "Discounted cost")
save_plot(3, [ x.CQ for x in state_vec], "Channel Quality")
save_plot(4, Network_util, "Network Utility")
save_plot(5, FOV_hand, "FOV availability of next segment")
save_plot(6, Freeze_prob, "Probabity of freeze")
#in the long run, how much time was spent in a frozen screen state