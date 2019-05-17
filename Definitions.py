import numpy as np
import math
# transition probabilities for channel quality
p = 0.5
q = 0.6
Up = np.matrix([[0,0],[1-q,q]])
W = np.matrix([[1-p,p],[0,0]])
Z_u = np.zeros(Up.shape)

# Number of tiles in a frame
T = 16
# length of single block
Q = 10
M = 4
# distribution function and the identity function
f_param = 0.5
def f_d(x,block_num = 1):
	return(math.exp(-f_param*x/block_num))

def F(x,block_num = 1):
	sum1 = sum([f_d(y+1,block_num) for y in range(x)])
	sum2 = sum([f_d(y+1,block_num) for y in range(T)])
	return(sum1/sum2)

def Identity(x,block_num=1):
	s = 0.95 - 0.05*block_num
	if F(x,block_num) >= s:
		return 1
	return 0