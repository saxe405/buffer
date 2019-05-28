__all__ = ["isValidAction", "isValidState"]
from Definitions import *

def Neighbours(action,s):						# if action a is taken in state a, what are the possible states it can end up in. Clearly for a given s and a, all the `obabilities should sum upto 1
	Neighbours = set()
	if not isValidAction(action,s):
		return Neighbours
	new_n1 = -1
	new_n2 = -1
	new_L  = -1
	new_N  = -1
	new_R  = -1
	if action == 1: #download to n1
		if s.R > 1:
			new_n1 = s.n1+1
			new_n2 = s.n2
			new_L  = s.L
			new_N  = s.N
			new_R  = s.R-1
		elif s.L > 0:
			new_n1 = s.n2
			new_n2 = M
			new_L  = s.L-1
			new_N  = s.N
			new_R  = Q			
		else:
			new_n1 = s.n2
			new_n2 = s.N
			new_L  = 0
			new_N  = 0
			new_R  = Q			
	elif action == 2:
		if s.R > 1:
			new_n1 = s.n1
			new_n2 = s.n2+1
			new_L  = s.L
			new_N  = s.N
			new_R  = s.R-1			
		elif s.L > 0:
			new_n1 = s.n2+1
			new_n2 = M
			new_L  = s.L-1
			new_N  = s.N
			new_R  = Q			
		else:
			new_n1 = s.n2+1
			new_n2 = s.N
			new_L  = 0
			new_N  = 0
			new_R  = Q			
	elif action == 3:
		if s.R > 1 and s.N<M-1:
			new_n1 = s.n1
			new_n2 = s.n2
			new_L  = s.L
			new_N  = s.N+1
			new_R  = s.R-1			
		elif s.R > 1 and s.N == M-1:
			new_n1 = s.n1
			new_n2 = s.n2
			new_L  = s.L+1
			new_N  = 0
			new_R  = s.R-1			
		elif s.R == 1 and s.N < M-1:
			if s.L > 1:
				new_n1 = s.n2
				new_n2 = M
				new_L  = s.L-1
				new_N  = s.N+1
				new_R  = Q				
			else:
				new_n1 = s.n2
				new_n2 = s.N+1
				new_L  = 0
				new_N  = 0
				new_R  = Q				
		else:
			new_n1 = s.n2
			new_n2 = M
			new_L  = s.L
			new_N  = 0
			new_R  = Q		
	if new_n1 >= 0:		
		for i in range(11):
			st = new_state(int(new_n1*0.1*i),int(new_n2*0.1*i),new_L,new_N,new_R,s.CQ)
			if isValidState(st):
				for j in range(len(CQs)):
					if CQ_matrix[s.CQ,CQs[j]] > 0:
						Neighbours.add(new_state(int(new_n1*0.1*i),int(new_n2*0.1*i),new_L,new_N,new_R,j))
	return Neighbours	

def Probability_matrix(a, s1, s2):								# probability that that the system moves to state s2 from s1 given the action a is taken 
	if not isValidAction(a,s1):
		return 0	
	new_st = new_state(s1.n1,s1.n2,s1.L,s1.N,s1.R,s1.CQ)
	#modify s1 to represent the changes that are going to happen for sure
	if a == 1:
		new_st = new_st._replace(n1=new_st.n1+1)
	elif a == 2:
		new_st = new_st._replace(n2=new_st.n2+1)
	else:
		new_st = new_st._replace(N=new_st.N+1)
		if new_st.N == M:
			new_st = new_st._replace(L=new_st.L+1)
			new_st = new_st._replace(N=0)
	if new_st.R == 1:
		new_st = new_st._replace(R=Q,n1=new_st.n2,n2=M)		
		if new_st.L > 0:
			new_st = new_st._replace(L=new_st.L-1)
		else:
			new_st = new_st._replace(n2=new_st.N, L = 0, N = 0)			
	else:
		new_st = new_st._replace(R=new_st.R-1)
	if not (new_st.L == s2.L and new_st.N == s2.N and new_st.R == s2.R):
		return 0

	prob = CQ_matrix[s1.CQ,s2.CQ]
	
	# if changes in n1 and n2 are not in approximately the same ratio, then it is not due to the head movement
	if new_st.n1 > 0.1*T and new_st.n2 > 0.1*T:
		if abs( s2.n1*1.0/new_st.n1 - s2.n2*1.0/new_st.n2) > 0.05*(new_st.n1-s2.n1)*1.0/new_st.n1: 
			return 0
		key = round((new_st.n1-s2.n1)*1.0/new_st.n1,2)
		if key not in head_movement_lookup_table:
			return 0
		prob = head_movement_lookup_table[key]*prob			
	return prob