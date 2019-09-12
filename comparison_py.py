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

def CQ_from_TP(val, max):
	if val < max/4:
		return 0
	elif val < 3*max/4:
		return 1
	return 2

def read_traces(trace_num):
	f=open("5G_traces/WiGig_"+ str(trace_num) + ".txt", "r")
	contents = f.read().split('\n')
	f_contents = [ float(x) for x in contents if x != '']	
	return [ CQ_from_TP(x,max(f_contents)) for x in f_contents]

def save_plot(plot_num, Performance_index, y_lab, file_name, x_lab = 'Time'):
	plt.figure(plot_num)
	fig, ax = plt.subplots()

	ax.plot([0.1*x for x in range(len(Performance_index))], [x[0] for x in Performance_index], 'r-', label = 'streaming',alpha=0.7 )
	ax.plot([0.1*x for x in range(len(Performance_index))], [x[1] for x in Performance_index], 'b-', label = 'short pred caching', alpha=0.7 )	
	ax.plot([0.1*x for x in range(len(Performance_index))], [x[2] for x in Performance_index], 'g-', label = 's+l pred caching', alpha=0.7 )		

	plt.xlabel(x_lab)
	plt.ylabel(y_lab)
	legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
	legend.get_frame().set_facecolor('#00FFCC')
	plt.tight_layout()
	plt.grid(True)
	if args.trace_analysis:
		file_name += "PA"
	plt.savefig("C:/Users/sapoorv/Downloads/CODE/python/VR Cache/"+ file_name +".png")

print("reading 5G traces")
Channel_path = read_traces(args.trace_number)

def Channel_quality_change(channel_quality, timer, trace_analysis = False):
	if trace_analysis:
		if timer > 1: # traces were taken every 500ms			
			return channel_quality		
		global Channel_path
		cq = Channel_path[0]
		Channel_path = Channel_path[1:]
		return cq

	return np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )

def Action_table_builder(book_name):
	data = xlrd.open_workbook(book_name)
	table = data.sheets()[0]
	ncols = table.ncols
	num_iterations = int((ncols-8)/2)

	Action_table = {}

	for i in range(table.nrows):
		vals = table.row_values(i)		
		s = new_state(int(vals[0]),int(vals[1]),int(vals[2]),int(vals[3]),int(vals[4]),int(vals[5]))
		a = (int(vals[ncols-1][0]), int(vals[ncols-1][2:]) )	
		Action_table[s] = a
	return Action_table

Action_table = Action_table_builder('Updated Utilities_1.0.xlsx')
Action_table_short_cache = Action_table_builder('Updated Utilities_short_term_1.0.xlsx')

print("Action tables ready")
t = 0
cachin_state = starting_State
streamin_state = s_state(T,Q,cachin_state.CQ)
short_cache_state = starting_State
Cost_vec = [[0,0,0]]
QoE_vec = [[0,0,0]]
Network_util = [[0,0,0]]
print("Starting with the initial phase")
#print(list(head_movement_lookup_table.values()))
while t < 1000:
	t+=1
	best_c_action = Action_table[cachin_state]
	best_c2_action = Action_table_short_cache[short_cache_state]

	channel_quality = cachin_state.CQ

	next_c_state = cache_state_without_NP_and_CQ(best_c_action,cachin_state)
	next_short_c_state = cache_state_without_NP_and_CQ(best_c2_action,short_cache_state) # we shouldn't need a new function here
	next_s_state = stream_state_without_NP_and_CQ(streamin_state)

	head_movement = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table.values())))], p=list(head_movement_lookup_table.values()))	
	next_channel_quality = np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )

	#print(best_c2_action, short_cache_state)
	Cost_vec.append([	Cost_vec[-1][0]*gamma + Networking_Cost_stream(streamin_state, next_s_state) ,
						Cost_vec[-1][1]*gamma + Networking_Cost_CQ(best_c2_action,short_cache_state.CQ),
						Cost_vec[-1][2]*gamma + Networking_Cost_CQ(best_c_action,cachin_state.CQ),
		])
	Network_util.append([ Network_util[-1][0]*0.8 + Network_utilization_stream(streamin_state,next_s_state),
						  Network_util[-1][1]*0.8 + Network_utilization(short_cache_state,best_c2_action)	,					  
						  Network_util[-1][2]*0.8 + Network_utilization(cachin_state,best_c_action), 
						  
		])
	
	#modify next states by applying the head movement transformations and channel quality variation
	streamin_state = stream_state_NP(next_s_state,head_movement,next_channel_quality)
	cachin_state = cache_state_NP( next_c_state,head_movement,next_channel_quality)
	short_cache_state = cache_state_NP( next_short_c_state,head_movement,next_channel_quality)

state_vec 	= [[streamin_state, short_cache_state, cachin_state]]
FOV_vec 	= [[FOV_available_stream(streamin_state), FOV_available(short_cache_state), FOV_available(cachin_state)]]
Cost_vec 	= [Cost_vec[-1]]
QoE_vec 	= [QoE_vec[-1]]
Network_util = [Network_util[-1]]
FOV_hand 	= [[ 	FOV_handover_stream(streamin_state), 
					FOV_handover(short_cache_state), 
					FOV_handover(cachin_state)
				]]

Freeze_prob = [[ 	Prob_screen_freeze_stream(streamin_state), 
					Prob_screen_freeze(short_cache_state), 
					Prob_screen_freeze(cachin_state)
				]]

print("starting with the simulation phase")
t = 0
while t < 500:
	t+=1

	best_c_action = Action_table[cachin_state]
	best_c2_action = Action_table_short_cache[short_cache_state]

	channel_quality = cachin_state.CQ
	next_c_state = cache_state_without_NP_and_CQ(best_c_action,cachin_state)	
	next_short_c_state = cache_state_without_NP_and_CQ(best_c2_action,short_cache_state) # we shouldn't need a new function here
	next_s_state = stream_state_without_NP_and_CQ(streamin_state)

	if t < 0:
		print(channel_quality)
		print(max_tiles_in_CQ(channel_quality))		
		print(cachin_state)
		print(best_c_action)
		print(max_tiles_in_CQ(cachin_state.CQ))
		print(streamin_state)
		print(max_tiles_in_CQ(streamin_state.CQ))
		print(next_s_state)

	head_movement = np.random.choice([float(x/100) for x in range(len(list(head_movement_lookup_table.values())))], p=list(head_movement_lookup_table.values()))	
	next_channel_quality = Channel_quality_change( channel_quality, cachin_state.R, trace_analysis= args.trace_analysis)

	Cost_vec.append([	Cost_vec[-1][0]*gamma + Networking_Cost_stream(streamin_state, next_s_state) ,
						Cost_vec[-1][1]*gamma + Networking_Cost_CQ(best_c2_action,short_cache_state.CQ),
						Cost_vec[-1][2]*gamma + Networking_Cost_CQ(best_c_action,cachin_state.CQ),
		])
	Network_util.append([ Network_util[-1][0]*0.8 + Network_utilization_stream(streamin_state,next_s_state),
						  Network_util[-1][1]*0.8 + Network_utilization(short_cache_state,best_c2_action)	,					  
						  Network_util[-1][2]*0.8 + Network_utilization(cachin_state,best_c_action), 
						  
		])

	FOV_hand.append([ 	FOV_handover_stream(next_s_state), 
						FOV_handover(next_short_c_state), 
						FOV_handover(next_c_state)
					])
	Freeze_prob.append([ 	Prob_screen_freeze_stream(next_s_state), 
							Prob_screen_freeze(next_short_c_state), 
							Prob_screen_freeze(next_c_state)
						])
	
	FOV_vec.append([	FOV_available_stream(next_s_state), 
						FOV_available(next_short_c_state), 
						FOV_available(next_c_state)
					])
	
	#modify next states by applying the head movement transformations and channel quality variation
	streamin_state = stream_state_NP(next_s_state,head_movement,next_channel_quality)
	cachin_state = cache_state_NP( next_c_state,head_movement,next_channel_quality)
	short_cache_state = cache_state_NP( next_short_c_state,head_movement,next_channel_quality)
		
	state_vec.append([streamin_state, short_cache_state, cachin_state])

#print(FOV_hand)
save_plot(1,FOV_vec, "FOV length available (segments))", "FOV_len_available" )
save_plot(2,Cost_vec, "Avg Networking Cost", "Average_networking_cost")
save_plot(3, [ [x[0].CQ,x[1].CQ, x[2].CQ] for x in state_vec], "Channel Quality", "channel_quality")
save_plot(4, Network_util, "Network Utilization", "Network_Utilization" )
save_plot(5, FOV_hand, "Pr( FOV of next segment available)", "FOV_availability")
save_plot(6, Freeze_prob, "Probabity of freeze","Probabity_of_freeze")
#in the long run, how much time was spent in a frozen screen state