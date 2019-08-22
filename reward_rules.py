__all__ = ["F"]
from Definitions import *

# cost incurred due to transition from s1 to s2 due to action a
def Cost_of_act_n_transact(s1,a,s2):
	spent = Networking_Cost_CQ(a,s1.CQ)
	gained = state_cost(s2) - state_cost(s1)
	return spent - gained

def Networking_Cost_CQ(action, quality):
	if action[0] == 0:
		return 0
	return action[1]*Cost_CQ_base[quality] # number of tiles dowloaded = action%10 * cost of downloading a single tile

def state_cost(s1): #cost of being in state s1
	return (1-F(s1.n1,1))*Reward_matrix_coeff[0] + (1-F(s1.n2,2))*Reward_matrix_coeff[1] + Reward_matrix_coeff[2]*sum([s1.L*(1-F(M,k+2))*1 for k in range(1,s1.L)]) + (1-F(s1.N,s1.L+3))*Reward_matrix_coeff[3]

def Reward_matrix(a, s1):				# for each state s1, if the user takes the action a then it gets the returned award
	networking_cost = -1*Networking_Cost_CQ(a,s1.CQ)
	no_video_cost = 0
	if a[0] == 0 or s1.CQ == 0:	
		no_video_cost = 0
	elif a[0] == 1:
		s1 = s1._replace(n1=s1.n1+ a[1])		

	elif a[0] == 2:
		s1 = s1._replace(n2=s1.n2+ a[1])
	
	elif a[0] == 3:
		s1 = s1._replace(N=s1.N+ a[1])	
		while s1.N >= M:
			s1 = s1._replace(N=s1.N - M, L = s1.L+1)
	
	no_video_cost = -1*state_cost(s1)

	qoe_cost = 0 						#interpret this later
	return networking_cost + no_video_cost + qoe_cost

	'''
	How are the values of penalties decided
	each tile of block 1 contributes 500*0.95/8 worth of penalty ~ 59
	each tile of block 2 contributes 200*0.95/8 worth of penalty ~ 24
	each tile of further blocks contribute 20*0.95/8 worth of penalty ~2

	This values are close to the neworking cost 
	'''