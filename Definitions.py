from collections import namedtuple
from collections import defaultdict
from channel import get_CQ_matrix, max_tiles_in_CQ
from channel import *
import argparse

parser = argparse.ArgumentParser(prog='cache_mdp')
parser.add_argument("--verbosity", "-v", help = "increase output verbosity", type = bool, default = True)
parser.add_argument("--penalty", '-p', help = "change scale of penalty( 75 - low, 1 - moderate, 150 - high", type = int, default = 160)
parser.add_argument("--L_max", '-L_max', help = "maximum number of long prediction blocks allowed", type = int, default = 5)
parser.add_argument("--video_group", '-vg', help = "video group 0 - static, 1 - regular, 2 - erratic", type = int, default = 1 )
parser.add_argument("--user_group", '-ug', help = "user group 0 - static, 1 - regular, 2 - erratic", type = int, default = 1 )
parser.add_argument("--trace_analysis", "-ta", help = "True: use traces for channel quality, False: use channel quality matrix", type  = bool, default = False )
parser.add_argument("--trace_number", "-tn", help = "which trace number to use for analysis 1:5", type = int, default = 5)
parser.add_argument("--pi0", "-pi0", help = "avg time spend in bad network state, 0-1", type = float, default = 0.2 )
parser.add_argument("--Mtrust", "-Mt", help = "level of trust you put on long term predictions", type = float, default = 0.80)
args = parser.parse_args()

CQ_matrix = get_CQ_matrix_pi(args.pi0)

Reward_matrix_coeff = [90,25,8,4]
#Reward_matrix_coeff = [args.penalty*x for x in Reward_matrix_coeff]

if args.verbosity:
    print("CQ Matrix built!")

Num_tiles_in_Video = 200                        # Number of tiles in a frame
def max_tiles(video_group = args.video_group):
    T = int(Num_tiles_in_Video*0.20/4) + 2                 #20 percent of total 200 tiles
    if video_group == 0:
        T = T-2     # 10 for static video
    elif video_group == 2:
        T = T+2     # 
    return T
    '''
    T = int(Num_tiles_in_Video*0.15)                 #20 percent of total 200 tiles
    if video_group == 0:
        T = 0.8*T     # 10 for static video
    elif video_group == 2:
        T = 1.2*T     #     
    return T
    '''
Q = 5                                            # length of single block, in terms of slot
M = int(max_tiles(video_group = args.video_group)*args.Mtrust)
longer_channel_stay = True
new_state = namedtuple("State", "n1 n2 L N R CQ")
Utility_table = defaultdict(lambda : -1000)
Utility_table_iterations = defaultdict(lambda: "")
Action_table_iterations = defaultdict(lambda: "")
Action_table = defaultdict(lambda : (-1,-1))

def starting_State_f(T = max_tiles()):
    return new_state(T,0,0,0,Q,2)

def f_d(x,block_num = 1, video_group = args.video_group):                # distribution function and the identity function
    T = max_tiles(video_group = video_group)
    r = T-2 #95 percent distributed over these tiles and remaining 0.05 over the remaining
    if x == 0 or x > T:
        return 0
    elif x <= r:
        return 0.95/r
    else:
        return 0.05/(T-r)

def F(x,block_num = 1, video_group = args.video_group):    
    val = 0.34*(x**(0.48))
    if video_group == 1:
        val = 0.23*(x**(0.59))
    elif video_group == 2:
        val = 0.20*(x**(0.62))
    return(min(val,1))
    '''
    val = 0.3*(x**(0.38))
    if video_group == 1:
        val = 0.26*(x**(0.4))
    elif video_group == 2:
        val = 0.21*(x**(0.44))
    return(min(val,1))
    '''
def Identity(x,block_num=1, video_group = args.video_group):
    s = 0.95 - 0.05*block_num
    if F(x,block_num, video_group = video_group) >= s:
        return 1
    return 0

def isValidAction(act, st, T = max_tiles()):                        # is a valid action from state s
    #if s.CQ == 0 and a != (0,0):            # can not download anything in state 0
        #return False    
    if act[0] == 0 and act[1] > 0:
        return False
    if act[1] > max_tiles_in_CQ(st.CQ,T):                # more than 5 tiles can be downloaded only if the network is in best state
        return False

    if act[0] == 1 and act[1] + st.n1 > T:
        return False
    if act[0] == 2 and act[1] + st.n2 > T:
        return False    
    return True

def isValidState(s, L_max = args.L_max, T = max_tiles()):
    if s.n1 < 0 or s.n1 > T:
        return False
    if s.n2 < 0 or s.n2 > T:
        return False
    if s.L < 0 or s.L > L_max:
        return False
    if s.N < 0 or s.N >= M or s.N > L_max*M:
        return False
    if s.R < 1 or s.R >Q:
        return False
    if s.CQ < 0 or s.CQ > CQ_matrix.shape[0]:
        return False
    return True

# is field of view available for next 1 second if the things go down, i.e. for the next 2*Q slots,
# we basically need availability of FOV for the first 2 blocks
def FOV_available(s, video_group = args.video_group): 
    buffer_len = Identity(s.n1,1,video_group = video_group) 
    if buffer_len > 0:
        buffer_len+= Identity(s.n2,2, video_group = video_group)
    return buffer_len

def Network_utilization(s,a,video_group = args.video_group):
    if s.CQ == 0:
        return 0
    return a[1]*100/max_tiles_in_CQ(s.CQ, max_tiles(video_group = video_group))    #max 5 can be supported by bandwidth but only a[1] amount used
    

def Prob_screen_freeze(s, video_group = args.video_group):
    return 1 - FOV_handover(s, video_group = video_group)

def FOV_handover(s, video_group = args.video_group):    
    val = F(s.n1, video_group = video_group)
    if val < 0.9:
        return 0
    return val

def FOV_quality_fn(n, video_group = args.video_group):
    return F(n, video_group = video_group)

def Utility_book_name(penalty,l_max,user_group,video_group, pi0, Mtrust):
    book_name = "Utilities/Updated Utilities_"
    book_name = book_name + str(penalty)+ "_" + str(l_max) + "_" + str(user_group)+ "_" + str(video_group) + "_" + str(pi0) + "_" + str(Mtrust) + ".xlsx"
    return book_name

def Utility_book_name_old(penalty,l_max,user_group,video_group):
    book_name = "Utilities/Updated Utilities_"
    book_name = book_name + str(penalty)+ "_" + str(l_max) + "_" + str(user_group)+ "_" + str(video_group) + ".xlsx"
    return book_name

gamma = 0.99                                 # discount factor of MDP

# (s,n) -> download n tiles in segment s where s = 1 (first short term segment),2 (second short term segment),3 (long term segments) and n <10. 
All_actions =  [(0,0)] 
#max_act = max([ max_tiles_in_CQ(c,max_tiles()) for c in range(CQ_matrix.shape[0])])

for n in range(1,5):
    All_actions += [(s,n) for s in range(1,3)]
    if args.L_max > 0:
        All_actions+= [(3,n)]