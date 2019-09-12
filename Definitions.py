import numpy as np
import math
from collections import namedtuple
from collections import defaultdict
import xlrd
from channel import *
import os
import argparse

parser = argparse.ArgumentParser(prog='cache_mdp')
parser.add_argument("--num_iters", "-ni", help = "number of value iterations to perform", type  = int, default = 2 )
parser.add_argument("--verbosity", "-v", help = "increase output verbosity", type = bool, default = False)
parser.add_argument("--penalty", '-p', help = "change scale of penalty( 0.75 - low, 1 - moderate, 1.5 - high", type = float, default = 1.0)
parser.add_argument("--collapse_CQ", '-CQ', help = "Reduce the number of channel states", type = bool, default = True)
parser.add_argument("--L_max", '-L_max', help = "maximum number of long prediction blocks allowed", type = int, default = 5)
parser.add_argument("--trace_analysis", "-ta", help = "True: use traces for channel quality, False: use channel quality matrix", type  = bool, default = False )
parser.add_argument("--trace_number", "-tn", help = "which trace number to use for analysis 1:5", type = int, default = 5)
args = parser.parse_args()

'''
if os.path.isfile('./CQ_matrix.xlsx'):
	CQ_matrix = read_from_disk("CQ_matrix.xlsx", collapse = args.collapse_CQ)
	noise = np.random.rand(CQ_matrix.shape[0],CQ_matrix.shape[1])
	for i in range(noise.shape[0]):
		noise[i] = noise[i]/sum(noise[i])
	CQ_matrix = noise	
else:
	CQ_matrix = channel_matrix(collapse = args.collapse_CQ)
'''

Reward_matrix_coeff = [90,20,1,0.1]
Reward_matrix_coeff = [args.penalty*x for x in Reward_matrix_coeff]

if args.verbosity:
	print("CQ Matrix built!")

Num_tiles_in_Video = 200						# Number of tiles in a frame
T = int(Num_tiles_in_Video*0.20/4) 				#20 percent of total 200 tiles
L_max = args.L_max				
Q = 5											# length of single block, in terms of slot
M = 3

new_state = namedtuple("State", "n1 n2 L N R CQ")
Utility_table = defaultdict(lambda : -1000)
Utility_table_iterations = defaultdict(lambda: "")
Action_table_iterations = defaultdict(lambda: "")
Action_table = defaultdict(lambda : (-1,-1))
starting_State = new_state(T,0,0,0,Q,2)

def f_d(x,block_num = 1):				# distribution function and the identity function
	r = T-2 #95 percent distributed over these tiles and remaining 0.05 over the remaining
	if x == 0 or x > T:
		return 0
	elif x <= r:
		return 0.95/r
	else:
		return 0.05/(T-r)

def F(x,block_num = 1):	
	val = 0.34*(x**(0.48))
	return(min(val,1))

def Identity(x,block_num=1):
	s = 0.95 - 0.05*block_num
	if F(x,block_num) >= s:
		return 1
	return 0

def isValidAction(a,s):						# is a valid action from state s
	#if s.CQ == 0 and a != (0,0):			# can not download anything in state 0
		#return False
	if a[1] > max_tiles_in_CQ(s.CQ):				# more than 5 tiles can be downloaded only if the network is in best state
		return False

	if a[0] == 1 and a[1] + s.n1 > T:
		return False
	if a[0] == 2 and a[1] + s.n2 > T:
		return False
	return True

def isValidState(s):
	if s.n1 < 0 or s.n1 > T:
		return False
	if s.n2 < 0 or s.n2 > T:
		return False
	if s.L < 0 or s.L > L_max:
		return False
	if s.N < 0 or s.N >= M:
		return False
	if s.R < 1 or s.R >Q:
		return False
	if s.CQ < 0 or s.CQ >= len(CQs):
		return False
	return True

# is field of view available for next 1 second if the things go down, i.e. for the next 2*Q slots,
# we basically need availability of FOV for the first 2 blocks
def FOV_available(s): 
	buffer_len = Identity(s.n1,1) 
	if buffer_len > 0:
		buffer_len+= Identity(s.n2,2)
	return buffer_len

def Network_utilization(s,a):
	if s.CQ == 0:
		return 0
	return a[1]*100/max_tiles_in_CQ(s.CQ)    #max 5 can be supported by bandwidth but only a[1] amount used
	

def Prob_screen_freeze(s):
	return 1 - F(s.n1)	

def FOV_handover(s):	
	return F(s.n1)
	

gamma = 0.99 								# discount factor of MDP

# (s,n) -> download n tiles in segment s where s = 1 (first short term segment),2 (second short term segment),3 (long term segments) and n <10. 
All_actions =  [(0,0)] 
for n in range(1,T):
	All_actions += [(s,n) for s in range(1,4)]