__all__ = ["F"]
from Definitions import *
def Reward_matrix(a, s1):				# for each state s1, if the user takes the action a then it gets the returned award
	networking_cost = -1*Cost_CQ[a][s1.CQ]
	no_video_cost = 0
	if a == 0 or s1.CQ == 0:
		no_video_cost = -1*(1-F(s1.n1,1))*50 - (1-F(s1.n2,2))*30 - sum([s1.L*(1-F(M,k+2))*1 for k in range(1,s1.L)]) - (1-F(s1.N,s1.L+3))*1
	elif a == 1:
		no_video_cost = -1*(1-F(s1.n1+1,1))*50 - (1-F(s1.n2,2))*30 - sum([s1.L*(1-F(M,k+2))*1 for k in range(1,s1.L)]) - (1-F(s1.N,s1.L+3))*1
	elif a == 2:
		no_video_cost = -1*(1-F(s1.n1,1))*50 - (1-F(s1.n2+1,2))*30 - sum([s1.L*(1-F(M,k+2))*1 for k in range(1,s1.L)]) - (1-F(s1.N,s1.L+3))*1
	elif a == 3:
		no_video_cost = -1*(1-F(s1.n1,1))*50 - (1-F(s1.n2,2))*30 - sum([s1.L*(1-F(M,k+2))*1 for k in range(1,s1.L)]) - (1-F(s1.N+1,s1.L+3))*1
	
	qoe_cost = 0 						#interpret this later
	return networking_cost + no_video_cost + qoe_cost