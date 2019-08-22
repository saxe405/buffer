from butools.mam import *
from Definitions import *
import numpy as np

# block matrix S using Up W and F
Si = np.zeros((2,2*(T+1))) #corresponding to zero tiles
for i in range(1,T+1):
	this_block = Up*Identity(i)
	Si = np.block([
    	 	[Si],
     		[this_block, np.zeros((2,2*T))               ]
 	 	])
Siprime = np.block([
			 [W, Up, np.zeros((2,2*(T-1)))] 
	])
for i in range(1, T):
	this_block = Up*(1-Identity(i))
	Siprime = np.block([
					[Siprime],
					[np.zeros((2,2*i)), W, this_block, np.zeros((2,2*(T-i-1)))]
				])
Siprime = np.block([
					[Siprime],
					[np.zeros((2,2*T)), W]
			])

S_iprime = ml.matrix(Siprime)
S_i = ml.matrix(Si)

shap = S_i.shape[0] # need this to extend S to M

Mmin1 = np.block([ 
			[np.zeros((shap,shap*(Q-1))), Siprime ],
			[np.zeros((shap*(Q-1),shap*Q))]
			])
M0 = np.block([ 
		[np.zeros((shap,shap*(Q-1))), Si ],	
	])
Mp1 = np.block([
			[np.zeros((shap, shap*Q))]
		])
for i in range(1,Q):
	M0 = np.block([
			[M0],
			[np.zeros((shap,shap*(i-1))), Siprime, np.zeros((shap,shap*(Q-i)))]
		])
	Mp1 = np.block([
			[Mp1],
			[np.zeros((shap,(i-1)*shap)), Si, np.zeros((shap,shap*(Q-i)))]
		])
# Mmin1 is B, M0 is L and Mp1 is F

B = ml.matrix(Mmin1)
L = ml.matrix(M0)
F = ml.matrix(Mp1)
L0 = ml.matrix(Mmin1)
#check if all row sums are equal to 1  == np.ones((Q*shap,1))
if [x.item(0,0) for x in (Mmin1.sum(axis=1) + M0.sum(axis = 1) + Mp1.sum(axis=1))] ==list([1.0 for x in range(shap*Q)]) == False:
	print("We have a problem!")


#R, G, U = QBDFundamentalMatrices (B, L, F, matrices="RGU")
#R1 = QBDFundamentalMatrices (B, L, F, "R")
#print(L0+R1*B)
#butools.verbose = True
'''
print(B + L*G + F*G**2)
print(F + R*L + R**2*B)
print(L + F*(-U).I*B - U)
'''
#pi0, R = QBDSolve (B, L, F, L0, prec = 1e-19)
#print("pi0= ", pi0)
#print("R= ", R)

#QBDStationaryDistr (pi0, R, 10)
