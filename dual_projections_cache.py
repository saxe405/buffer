__all__ = ["Identity"]
from Definitions import *
from butools.mam import *
#import resource
def matrix_F():
	F_3 = np.zeros((Up.shape[0]*(M-1),Up.shape[0]*M))
	F_3 = np.block([
			[F_3],
			[Up, np.zeros((Up.shape[0],(M-1)*Up.shape[0]))]
		])
	F_2 = np.zeros((F_3.shape[0], Q*F_3.shape[0] ))
	for i in range(1,Q):
		F_2 = np.block([
			[F_2],
			[ np.zeros(( F_3.shape[0], (i-1)*F_3.shape[0])), F_3, np.zeros((F_3.shape[0], (Q-i)*F_3.shape[0] ))]
			])
	F_1 = np.block([
			[Identity(0,2)*F_2, np.zeros((F_2.shape[0], (T)*F_2.shape[0])) ]
		])
	for i in range(1,T+1):
		#print(F_1.shape)
		#print(i* F_2.shape[0] + F_2.shape[0] + (T-i)*F_2.shape[0])		
		F_1 = np.block([
			[F_1],
			[np.zeros(( F_2.shape[0], i*F_2.shape[0])), Identity(i,2)*F_2, np.zeros((F_2.shape[0], (T-i)*F_2.shape[0])) ]
			])
	F = np.block([
			[Identity(0,1)*F_1, np.zeros((F_1.shape[0], (T)*F_1.shape[0])) ]
		])
	for i in range(1,T+1):
		F = np.block([
			[F],
			[np.zeros(( F_1.shape[0], i*F_1.shape[0])), Identity(i,1)*F_1, np.zeros((F_1.shape[0], (T-i)*F_1.shape[0])) ]
			])
	return F		

def matrix_L():
	J_2p = np.block([
		[W, np.zeros((W.shape[0], W.shape[0]*(M-1)))]
		])
	for i in range(1,M):
		J_2p = np.block([
			[J_2p],
			[ np.zeros((W.shape[0],W.shape[0]*i)), W, np.zeros((W.shape[0], W.shape[0]*(M-1-i)))]
			])
	J_1p = np.zeros((J_2p.shape[0], J_2p.shape[0]*Q))
	for i in range(1,Q):
		J_1p = np.block([
			[J_1p],
			[ np.zeros((J_2p.shape[0],J_2p.shape[0]*(i-1))), J_2p, np.zeros((J_2p.shape[0], J_2p.shape[0]*(Q-i)))]
			])
	J_1 = np.block([
		[J_1p, np.zeros((J_1p.shape[0], J_1p.shape[0]*(T)))]
		])
	for i in range(1,T+1):
		J_1 = np.block([
			[J_1],
			[ np.zeros((J_1p.shape[0],J_1p.shape[0]*i)), J_1p, np.zeros((J_1p.shape[0], J_1p.shape[0]*(T-i)))]
			])
	#second part
	J_3 = np.block([
		[np.zeros(Up.shape), Up, np.zeros((Up.shape[0], Up.shape[0]*(M-2)))]
		])
	for i in range(1,M-1):
		J_3 = np.block([
			[J_3],
			[np.zeros(( Up.shape[0],Up.shape[0]*(i+1))), Up,  np.zeros(( Up.shape[0],Up.shape[0]*(M-i-2))) ]
			])
	J_3 = np.block([
			[J_3],
			[np.zeros((Up.shape[0], Up.shape[0]*M))]
				])
	J_2 = np.zeros((J_3.shape[0], J_3.shape[0]*Q))
	for i in range(1,Q):	
		J_2 = np.block([
			[J_2],
			[ np.zeros((J_3.shape[0],J_3.shape[0]*(i-1))), J_3, np.zeros((J_3.shape[0], J_3.shape[0]*(Q-i)))]
			])
	L_3p = np.block([
		[Up, np.zeros((Up.shape[0], Up.shape[0]*(M-1)))]
		])
	for i in range(1,M):
		L_3p = np.block([
			[L_3p],
			[ np.zeros((Up.shape[0],Up.shape[0]*i)), Up, np.zeros((Up.shape[0], Up.shape[0]*(M-1-i)))]
			])
	L_2p = np.zeros((L_3p.shape[0], L_3p.shape[0]*Q))
	for i in range(1,Q):
		L_2p = np.block([
			[L_2p],
			[ np.zeros((L_3p.shape[0],J_3.shape[0]*(i-1))), L_3p, np.zeros((L_3p.shape[0], L_3p.shape[0]*(Q-i)))]
			])
	L_1p = np.block([
		[L_2p, np.zeros((L_2p.shape[0], L_2p.shape[0]*(T)))]
		])
	for i in range(1,T+1):
		L_1p = np.block([
			[L_1p],
			[ np.zeros((L_2p.shape[0],L_2p.shape[0]*i)), L_2p, np.zeros((L_2p.shape[0], L_2p.shape[0]*(T-i)))]
			])
	L_1 = np.block([
			[Identity(0,2)*J_2, (1-Identity(0,2))*L_2p, np.zeros((J_2.shape[0],J_2.shape[0]*(T-1)))]
		])
	for i in range(1,T):
		L_1 = np.block([
			[L_1],
			[ np.zeros((J_2.shape[0], J_2.shape[0]*i)), Identity(i,2)*J_2 , (1-Identity(i,2))*L_2p, np.zeros((J_2.shape[0], J_2.shape[0]*(T-i-1)))]
			])
	L_1 = np.block([
			[L_1],
			[np.zeros((J_2.shape[0], J_2.shape[0]*T)), Identity(T,2)*J_2]
		])
	L = np.block([
			[Identity(0,1)*L_1 + J_1, (1-Identity(0,1))*L_1p, np.zeros((J_1.shape[0],J_1.shape[0]*(T-1)))]
		])
	for i in range(1,T):
		#print(i, T, J_1.shape, L_1.shape)
		L = np.block([
			[L],
			[ np.zeros((J_1.shape[0], J_1.shape[0]*i)), Identity(i,1)*L_1+J_1 , (1-Identity(i,1))*L_1p, np.zeros((J_1.shape[0], J_1.shape[0]*(T-i-1)))]
			])
	L = np.block([
			[L],
			[np.zeros((J_1.shape[0], J_1.shape[0]*T)), Identity(T,1)*L_1+J_1]
		])
	return L

def matrix_BIJ(C, j, index):
	#j,M - for 1,3,4
	#j+1,M - for 2
	row_num = j
	if index == 2:
		row_num+=1
	if j == T:
		return np.zeros((C.shape[0]*(T+1), C.shape[0]*(T+1)))
	Bij = np.block([ 			
			[np.zeros((C.shape[0],C.shape[0]*(M-1))), C, np.zeros((C.shape[0],C.shape[0]*(T+1-M)))],
			])
	if row_num !=0:
	#	print( Bij.shape, C.shape, T, row_num)		
		Bij = np.block([
			[np.zeros((C.shape[0]*(row_num), C.shape[0]*(T+1)))],		
			[Bij]
			])		
	Bij = np.block([
			[Bij],					
			[np.zeros((C.shape[0]*(T-row_num), C.shape[0]*(T+1)))]
		])		
	return Bij


def matrix_B():
	C_6 = np.block([
		[W, np.zeros((W.shape[0], W.shape[0]*(M-1)))]
		])
	for i in range(1,M):
		C_6 = np.block([
			[C_6],
			[np.zeros((W.shape[0], W.shape[0]*i)), W, np.zeros((W.shape[0], W.shape[0]*(M-i-1)))]
			])
	C_5 = np.block([
			[np.zeros((C_6.shape[0], (Q-1)*C_6.shape[0])), C_6],
			[np.zeros((C_6.shape[0]*(Q-1), C_6.shape[0]*Q))]
		])
	C_4 = np.block([
		[Up, np.zeros((Up.shape[0], Up.shape[0]*(M-1)))]
		])
	for i in range(1,M):
		C_4 = np.block([
			[C_4],
			[np.zeros((Up.shape[0], Up.shape[0]*i)), Up, np.zeros((Up.shape[0], Up.shape[0]*(M-i-1)))]
			])
	C_3 = np.block([
			[np.zeros((C_4.shape[0], (Q-1)*C_4.shape[0])), C_4],
			[np.zeros((C_4.shape[0]*(Q-1), C_4.shape[0]*Q))]
		])		
	C_2 = np.zeros((Up.shape[0], M*Up.shape[0]))
	for i in range(1,M):
		C_2 = np.block([
			[np.zeros((Up.shape[0],(M-i)*Up.shape[0] )), Up, np.zeros((Up.shape[0], Up.shape[0]*(i-1)))],
			[C_2]
			])
	C_1 = np.block([
			[np.zeros((C_2.shape[0], (Q-1)*C_4.shape[0])), C_2],
			[np.zeros((C_2.shape[0]*(Q-1), C_2.shape[0]*Q))]
		])
	B = 0
	for i in range(T+1):		
		this_row = Identity(i,1)*Identity(0,2)*matrix_BIJ(C_1,0,1) + Identity(i,1)*(1-Identity(0,2))*matrix_BIJ(C_3,0,2)+ (1-Identity(i,1))*matrix_BIJ(C_3,0,3) + matrix_BIJ(C_5,0,4)
		for j in range(1,T+1):
			new_block = Identity(i,1)*Identity(j,2)*matrix_BIJ(C_1,j,1) + Identity(i,1)*(1-Identity(j,2))*matrix_BIJ(C_3,j,2)+ (1-Identity(i,1))*matrix_BIJ(C_3,j,3) + matrix_BIJ(C_5,j,4)
			this_row = np.block([
				[this_row, new_block]
				])
		if i == 0:
			B = this_row
		else:
			B = np.block([
				[B],
				[this_row]
				])
	return B

def matrix_D_i(CQ,i,index):
	if index == 1:
		i-=1
	if i < 0:
		return np.zeros((CQ.shape[0]*M, CQ.shape[0]*M))
	
	D_i = np.block([
			[np.zeros((CQ.shape[0]*i, CQ.shape[0]*M))],
			[CQ, np.zeros((CQ.shape[0],CQ.shape[0]*(M-1)))]
			])
	D_i = np.block([
			[D_i],
			[np.zeros((CQ.shape[0]*(M-i-1), CQ.shape[0]*M))]			
			])
	return D_i

def matrix_C_i(CQ,i,index):
	D_i = matrix_D_i(CQ,i,index)
	#print(Q, D_i.shape)
	#print(CQ,i,index)
	C_i = np.block([
		[ np.zeros((D_i.shape[0], D_i.shape[0]*(Q-1))), D_i ],
		[ np.zeros((D_i.shape[0]*(Q-1), D_i.shape[0]*Q))]
	])
	return C_i

def matrix_BIJ_p(j,index):
	row_num = j
	max_column = M-1
	CQ=Up
	C_i  = matrix_C_i(CQ,0,index)
	Bij_p = C_i
	if index == 2:
		row_num -=1
	if index == 1:
		max_column = M		
		Bij_p  = np.zeros(C_i.shape)
	if index == 4:
		CQ = W

	#print(Bij_p.shape)
	if row_num <0:
		return np.zeros(((T+1)*C_i.shape[0], (T+1)*C_i.shape[0]))

	for j in range(1,max_column+1):
		Bij_p = np.block([
			[Bij_p, matrix_C_i(CQ,j,index)]
			])
	Bij_p = np.block([
			[Bij_p, np.zeros((C_i.shape[0], (T-max_column)*C_i.shape[0]))]
			])
	#print(row_num, T, j)
	Bij_p = np.block([
		[np.zeros(((row_num)*C_i.shape[0], (T+1)*C_i.shape[0]))],
		[Bij_p],
		[np.zeros(((T-row_num)*C_i.shape[0], (T+1)*C_i.shape[0]))]
		])
	return Bij_p


def matrix_L0():
	L0 = 0
	for i in range(T+1):
		#print(matrix_BIJ_p(0,3).shape)
		this_row = Identity(i,1)*Identity(0,2)*matrix_BIJ_p(0,1) + Identity(i,1)*(1-Identity(0,2))*matrix_BIJ_p(0,2)+ (1-Identity(i,1))*matrix_BIJ_p(0,3) + matrix_BIJ_p(0,4)
		
		for j in range(1,T+1):
			new_block = Identity(i,1)*Identity(j,2)*matrix_BIJ_p(j,1) + Identity(i,1)*(1-Identity(j,2))*matrix_BIJ_p(j,2)+ (1-Identity(i,1))*matrix_BIJ_p(j,3) + matrix_BIJ_p(j,4)
			this_row = np.block([
				[this_row,new_block]
				])
		if i == 0:
			L0 = this_row
		else:
			L0 = np.block([
				[L0],
				[this_row]
				])
	return L0

F = matrix_F()
B =matrix_B()
L = matrix_L()


R, G, U = QBDFundamentalMatrices (B, L, F, matrices="RGU")
#L_0 = matrix_L0()
#pi0, R = QBDSolve (B, L, F, L0)
#print(matrix_D_i(Up,0,2))
#print(matrix_BIJ_p(1,3).shape)
