__all__ = ["isValidAction", "isValidState", "head_movement"]
from Definitions import *
from head_movements import *

T = max_tiles()
head_movement_matrix = np.zeros((T+1,T+1))
for den in range(1,T+1):
    for num in range(den+1):
        head_movement_matrix[den][num] = head_movement(num,den, user_group = args.user_group)

def Neighbours(action,s, T = max_tiles(), L_max = args.L_max):                        # if action a is taken in state a, what are the possible states it can end up in. Clearly for a given s and a, all the `obabilities should sum upto 1
    Neighbours = set()
    if not isValidAction(action,s, T = T):
        return Neighbours
    new_n1 = -1
    new_n2 = -1
    new_L  = -1
    new_N  = -1
    new_R  = -1
    
    if action[0] == 1 or action[0] == 0: #download to n1 or download nothing as 0%10 = 0
        if s.R > 1:
            new_n1 = s.n1 + action[1]
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
    elif action[0] == 2:
        if s.R > 1:
            new_n1 = s.n1
            new_n2 = s.n2 + action[1]
            new_L  = s.L
            new_N  = s.N
            new_R  = s.R-1            
        elif s.L > 0:
            new_n1 = s.n2 + action[1]
            new_n2 = M
            new_L  = s.L-1
            new_N  = s.N
            new_R  = Q            
        else:
            new_n1 = s.n2+action[1]
            new_n2 = s.N
            new_L  = 0
            new_N  = 0
            new_R  = Q            
    elif action[0] == 3:
        if s.R > 1 and s.N + action[1] <= M-1:
            new_n1 = s.n1
            new_n2 = s.n2
            new_L  = s.L
            new_N  = s.N+action[1]
            new_R  = s.R-1            
        elif s.R > 1 and s.N + action[1] > M-1:
            new_n1 = s.n1
            new_n2 = s.n2
            new_L  = s.L+1
            new_N  = s.N + action[1] - M
            new_R  = s.R-1            
        elif s.R == 1 and s.N + action[1] <= M-1:
            if s.L > 1:
                new_n1 = s.n2
                new_n2 = M
                new_L  = s.L-1
                new_N  = s.N + action[1]
                new_R  = Q                
            else:
                new_n1 = s.n2
                new_n2 = s.N + action[1]
                new_L  = 0
                new_N  = 0
                new_R  = Q                
        else:
            new_n1 = s.n2
            new_n2 = M
            new_L  = s.L
            new_N  = s.N + action[1] - M
            new_R  = Q
        while new_N >= M:
            new_L +=1
            new_N -=M    

    if new_n1 ==0 and isValidState(new_state(0,new_n2,new_L,new_N,new_R,s.CQ), L_max = L_max, T=T):
        if not longer_channel_stay or new_R == Q:
            for j in range(len(CQs)):
                Neighbours.add(new_state(0,new_n2,new_L,new_N,new_R,j))
        else:
            Neighbours.add(new_state(0,new_n2,new_L,new_N,new_R,s.CQ)) 
    elif new_n1 > 0:
        for n1 in range(new_n1+1):        
            st = new_state(n1,new_n2,new_L,new_N,new_R,s.CQ)
            if isValidState(st, L_max = L_max, T = T):
                if not longer_channel_stay or new_R == Q:
                    for j in range(len(CQs)):
                        if n1 >= new_n2:
                            Neighbours.add(new_state(n1,new_n2,new_L,new_N,new_R,j))                            
                        else:
                            ratio = n1*1.0/new_n1
                            Neighbours.add(new_state(n1,int(new_n2*ratio),new_L,new_N,new_R,j))
                else:
                    if n1 >= new_n2:
                        Neighbours.add(new_state(n1,new_n2,new_L,new_N,new_R,s.CQ))                            
                    else:
                        ratio = n1*1.0/new_n1
                        Neighbours.add(new_state(n1,int(new_n2*ratio),new_L,new_N,new_R,s.CQ))
    return Neighbours

def Probability_matrix(a, s1, s2, T = max_tiles()):                                # probability that that the system moves to state s2 from s1 given the action a is taken 
    
    if not isValidAction(a,s1, T = T):
        return 0
    new_st = new_state(s1.n1,s1.n2,s1.L,s1.N,s1.R,s1.CQ)
    #modify s1 to represent the changes that are going to happen for sure
    if a[0] == 1 or a[0] == 0:
        new_st = new_st._replace(n1=new_st.n1+ a[1])
    elif a[0] == 2:
        new_st = new_st._replace(n2=new_st.n2+ a[1])
    else:
        new_st = new_st._replace(N=new_st.N+ a[1])
        while new_st.N >= M:
            new_st = new_st._replace(L=new_st.L+1,N=new_st.N-M)
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
    
    if longer_channel_stay and new_st.R != Q and new_st.CQ != s2.CQ:
        return 0
    
    prob = 1
    if new_st.R == Q:
        prob = CQ_matrix[new_st.CQ,s2.CQ]
    if new_st.n1 == 0:
        if s2.n1 !=0 or new_st.n2 != s2.n2:
            return 0
        return prob
    elif new_st.n2 == 0:
        if s2.n2 != 0:
            return 0
        else:
            prob *= head_movement_matrix[new_st.n1][s2.n1]
            return prob
    else:
        ratio = 1.0*s2.n1/new_st.n1
        # the first part checks if both n1 and n2 decreased in the same ratio
        # the second part checks if n2 is not large enough to have a look at ratio 2
        if int(ratio*new_st.n2) == s2.n2 or ( s2.n1 >= s2.n2 and new_st.n2 == s2.n2):
            prob*=head_movement_matrix[new_st.n1][s2.n1]
            return prob
        print(ratio1, ratio2)
        print(s2, new_st)
    # if changes in n1 and n2 are not in approximately the same ratio, then it is not due to the head movement
#    if new_st.n1 > 0.1*T and new_st.n2 > 0.1*T:
#        if abs( s2.n1*1.0/new_st.n1 - s2.n2*1.0/new_st.n2) > 0.05*(new_st.n1-s2.n1)*1.0/new_st.n1: 
#            return 0
#        key = round((new_st.n1-s2.n1)*1.0/new_st.n1,2)
#        if key not in head_movement_lookup_table:
#            return 0
#        prob = head_movement_lookup_table[key]*prob            
    return 0

#this function returns the next state in which the system will be if we do not care about the channel quality and the head movements
#it is required to have a fair comparison across different streaming methods therefore, the stochastic parts will be varied 
#consistenly across all the methods
#that is, the stochastic nature will be kept common amongst the mechanisms
def cache_state_without_NP_and_CQ(a,s1, T = max_tiles()):
    if not isValidAction(a,s1, T = T):
        return 0
    new_st = new_state(s1.n1,s1.n2,s1.L,s1.N,s1.R,s1.CQ)
    #modify s1 to represent the changes that are going to happen for sure
    if a[0] == 1 or a[0] == 0:
        new_st = new_st._replace(n1=new_st.n1+ a[1])
    elif a[0] == 2:
        new_st = new_st._replace(n2=new_st.n2+ a[1])
    else:
        new_st = new_st._replace(N=new_st.N+ a[1])
        while new_st.N >= M:
            new_st = new_st._replace(L=new_st.L+1,N=new_st.N-M)
    if new_st.R == 1:
        new_st = new_st._replace(R=Q,n1=new_st.n2,n2=M)        
        if new_st.L > 0:
            new_st = new_st._replace(L=new_st.L-1)
        else:
            new_st = new_st._replace(n2=new_st.N, L = 0, N = 0)            
    else:
        new_st = new_st._replace(R=new_st.R-1)
    return new_st

#given the new state from the above function, modify to account for head movement
#stochasticity of head movement is handled seperately

def cache_state_NP(s,head_movement, Channel):
    s = s._replace(n1 = int(s.n1*head_movement))
    if not longer_channel_stay or s.R == Q:
        s = s._replace(CQ = Channel)
    if s.n2 > s.n1:        
        s = s._replace(n2 = int(s.n2*head_movement))
    return s