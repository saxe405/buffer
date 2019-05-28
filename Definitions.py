import numpy as np
import math
from collections import namedtuple
from collections import defaultdict

p = 0.5										# transition probabilities for channel quality
q = 0.5
Up = np.matrix([[0,0],[1-q,q]])
W = np.matrix([[1-p,p],[0,0]])

CQ_matrix = np.matrix([[1-p,p,0,0,0],[1-q,q/2,q/2,0,0],[0,1-q,q/2,q/2,0],[0,0,1-q,q/2,q/2],[0,0,0,1-q,q]])
Z_u = np.zeros(Up.shape)

Num_tiles_in_Video = 200						# Number of tiles in a frame
T = int(Num_tiles_in_Video*0.20/4) 				#20 percent of total 200 tiles
L_max = 10										
Q = 5											# length of single block, in terms of slot
M = 3

new_state = namedtuple("State", "n1 n2 L N R CQ")
Utility_table = defaultdict(lambda : 0)
Action_table = defaultdict(lambda : -1)
starting_State = new_state(0,0,0,0,Q,1)

def f_d(x,block_num = 1):				# distribution function and the identity function
	r = T-2 #95 percent distributed over these tiles and remaining 0.05 over the remaining
	if x == 0 or x > T:
		return 0
	elif x <= r:
		return 0.95/r
	else:
		return 0.05/(T-r)

def F(x,block_num = 1):
	sum1 = sum([f_d(y+1,block_num) for y in range(x+1)])
	sum2 = sum([f_d(y+1,block_num) for y in range(T+1)])
	return(sum1/sum2)

def Identity(x,block_num=1):
	s = 0.95 - 0.05*block_num
	if F(x,block_num) >= s:
		return 1
	return 0
## specific to MDP

def isValidAction(a,s):						# is a valid action from state s
	if s.CQ == 0 and a != 0:
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

gamma = 0.99 								# discount factor of MDP

All_actions =  [
	0, 										# do not download anything
	1, 										# download short term index 1
	2, 										# download short term index 2
	3, 										# download long term in the L+1 index
	#4, 									# serve the current FOV, i.e. don't build cache, user moved head.
]

CQs = [ x for x in range(5) ]				#channel qualities total 5 for now, for each state we have a corresponding cost of downloading

Cost_CQ = [[ 0 for x in range(len(CQs))] ] 	#0 cost for not downloading anything
Cost_CQ = Cost_CQ + [ [ 0,1,3,10,100 ], [0,0.5,1.5,8,80], [0,0.3,1.4,2,10], [0,0.2,1.3,3,4] ]						# these costs are limited to networking cost, we need to add the penalty of seeing a blank screen

theta_p0 = 4
theta_y0 = 2
theta_r0 = 1
theta_p_range = int(120/theta_p0)
theta_y_range = int(20/theta_y0)
theta_r_range = int(10/theta_r0)

def head_movement_probability(theta_p, theta_y, theta_r):						# probability that the user moves his head with this much angle for each dimension (pitch, yaw, roll)
	prob_1 = 0.85
	prob_2 = 0.90
	prob_3 = 0.98
	if theta_p != 0:
		prob_1 = (1-prob_1)/theta_p_range
	if theta_y != 0:
		prob_2 = (1-prob_2)/theta_y_range
	if theta_r != 0:
		prob_3 = (1-prob_3)/theta_r_range
	return prob_1*prob_2*prob_3

def head_movement(ratio_n):
	prob = 0
	for i in range(theta_p_range):
		for j in range(theta_y_range):
			for k in range(theta_r_range):
				if abs((1-i*theta_p0)*(1-j*theta_y0)*(1-k*theta_r0) - ratio_n) < 1.05*ratio_n:
					prob += head_movement_probability(i*theta_p0,j*theta_y0,k*theta_r0)
	return prob

print("Building the head movement lookup table: ")
head_movement_lookup_table = { 0 : head_movement(0)}
for i in range(1,100):
	head_movement_lookup_table[i/100] = head_movement(i/100)