'''
model the streaming methodology
that is only the next segment is streamed and corrected based 
on the head movement probabilites 
therefore in the state space, we just need to keep track of n1, CQ and R
to have a reasonable comparison
'''
from collections import namedtuple
from channel import *
from Definitions import *

s_state = namedtuple("Streaming_State", "n1 R CQ")

# the algo has to ensure that F(n1) is more than 0.99 something and download happens when the CQ is not 0
def stream_state_without_NP_and_CQ(s):
	max_tiles = max_tiles_in_CQ(s.CQ)
	new_state = s_state(s.n1,s.R,s.CQ)
	new_state = new_state._replace(R=new_state.R-1)
	if new_state.R == 0:
		new_state = new_state._replace(R=Q, n1=0)

	new_state = new_state._replace(n1=max(s.n1+max_tiles,T))
	return new_state

def stream_state_NP(state,head_movement,Channel):
	state._replace(n1 = int(state.n1*head_movement), CQ = Channel )
	return state

def Networking_Cost_stream(old_state,new_state):
	num_tiles_downloaded = new_state.n1 - old_state.n1 #this does not have head movements modification so it is okay
	return num_tiles_downloaded*Cost_CQ_base[old_state.CQ]

def Network_utilization_stream(old_state,new_state):
	num_tiles_downloaded = new_state.n1 - old_state.n1 #this does not have head movements modification so it is okay
	if old_state.CQ == 0:
		if num_tiles_downloaded !=0:
			raise Exception("can not download tiles when channel state is 0")
		return 0
	return num_tiles_downloaded/max_tiles_in_CQ(old_state.CQ)

def Prob_screen_freeze_stream(s):
	return 1 - F(s.n1)	

def FOV_handover_stream(s):	
	return F(s.n1)

def FOV_available_stream(s): 
	buffer_len = 0.5*Identity(s.n1,1) 
	return buffer_len