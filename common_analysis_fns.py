import xlrd
#__all__ = ["Identity", "Neighbours", "Probability_matrix"]
from Definitions import *
from transition_rules import cache_state_without_NP_and_CQ
from transition_rules import cache_state_NP
import numpy as np
from reward_rules import Networking_Cost_CQ
from streaming import Networking_Cost_stream, Network_utilization_stream, stream_state_without_NP_and_CQ, stream_state_NP, FOV_available_stream,FOV_handover_stream,Prob_screen_freeze_stream
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

def Channel_quality_change(channel_quality, timer, trace_analysis = False):
	if trace_analysis:
		if timer > 1: # traces were taken every 500ms			
			return channel_quality		
		global Channel_path
		cq = Channel_path[0]
		Channel_path = Channel_path[1:]
		return cq

	return np.random.choice([x for x in range(CQ_matrix.shape[0])], p=[CQ_matrix[channel_quality][0,x] for x in range(CQ_matrix.shape[0])] )

def next_channel_quality_fn(channel_quality, channel_vector, timer):
    if timer > 1:
        return [ channel_quality, channel_vector]
    return [channel_vector[0], channel_vector[1:]]
    
def Action_table_builder(book_name):    
    data = xlrd.open_workbook(book_name)
    table = data.sheets()[0]
    ncols = table.ncols
    #num_iterations = int((ncols-8)/2)
    
    Action_table = {}
    for i in range(table.nrows):
        vals = table.row_values(i)
        s = new_state(int(vals[0]),int(vals[1]),int(vals[2]),int(vals[3]),int(vals[4]),int(vals[5]))
        a = (int(vals[ncols-1][0]), int(vals[ncols-1][2:]))
        Action_table[s] = a
    return Action_table

def best_action_from_tables(Action_table, states, scenario):
	return Action_table[scenario][states[scenario]]

def Networking_Cost(scenario, cachin_states, next_states, Action_table):
	if scenario == "streaming":
		return Networking_Cost_stream(cachin_states["streaming"], next_states["streaming"] )
	
	channel_quality = cachin_states[ "both_predictions" ].CQ		
	return Networking_Cost_CQ( best_action_from_tables(Action_table,cachin_states, scenario), channel_quality)

def Network_Utilization_local(scenario, cachin_states,next_states, Action_table):
	if scenario == "streaming":
		return Network_utilization_stream(cachin_states["streaming"],next_states["streaming"])
	
	return Network_utilization(cachin_states[scenario], best_action_from_tables(Action_table,cachin_states, scenario))

def cache_state_local(scenario, cachin_states, Action_table, video_group = args.video_group):
	if scenario == "streaming":
		return stream_state_without_NP_and_CQ(cachin_states["streaming"], video_group = video_group)

	return cache_state_without_NP_and_CQ(best_action_from_tables(Action_table, cachin_states, scenario), cachin_states[scenario], T = max_tiles(video_group = video_group))

def next_state_local(scenario, head_movement, next_channel_quality, next_states):
	if scenario == "streaming":
		return stream_state_NP( next_states["streaming"], head_movement,next_channel_quality)
	return cache_state_NP( next_states[scenario], head_movement, next_channel_quality)

def FOV_available_local(scenario, cachin_states, video_group = args.video_group):
	if scenario == "streaming":
		return FOV_available_stream(cachin_states["streaming"], video_group = video_group)
	return FOV_available(cachin_states[scenario], video_group = video_group)

def FOV_handover_local(scenario,cachin_states, video_group = args.video_group):
	if scenario == "streaming":
		return FOV_handover_stream(cachin_states["streaming"], video_group = video_group)
	return FOV_handover(cachin_states[scenario], video_group = video_group)

def Freeze_prob_local(scenario,cachin_states, video_group = args.video_group):
	if scenario == "streaming":
		return Prob_screen_freeze_stream(cachin_states["streaming"], video_group = video_group)
	return Prob_screen_freeze(cachin_states[scenario], video_group = video_group)