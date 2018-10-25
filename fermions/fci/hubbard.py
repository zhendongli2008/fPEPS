#==================================================================
# Optimized H*Vec for Hubbard Model
#==================================================================
import numpy
from fci import bits

def genHVec(vec,info):
   return genHVec1(vec,info)+genHVec2(vec,info)

def genHVec1(vec,info):
   debug = False 
   if debug: print '[genHVec1]: 1e terms'
   [k,ndim,hlst,t,U]=info
   sigma = numpy.zeros(ndim)
   for idx in range(ndim):
      # H1: h[p,q]ap^+aq
      for i,j in hlst:
         pqlst = [[2*i,2*j],[2*i+1,2*j+1],[2*j,2*i],[2*j+1,2*i+1]]
         for p,q in pqlst:
            res1 = bits.action(idx,q,0)
            if res1 == None: continue
            idx1,sgn1 = res1
      	    res2 = bits.action(idx1,p,1)
            if res2 == None: continue
            idx2,sgn2 = res2
            sigma[idx2] += t*sgn1*sgn2*vec[idx] 
   return sigma

def genHVec2(vec,info):
   debug = False 
   if debug: print '[genHVec2]: 2e terms'
   [k,ndim,hlst,t,U]=info
   sigma = numpy.zeros(ndim)
   for idx in range(ndim):
      # V2: v[p,q,r,s]ap^+ aq^+ ar as 
      # U*aA^+*aA*aB^+*aB = U*aA^+*aB^+*aB*aA
      for i in range(k):
	 s = 2*i
         res1 = bits.action(idx,s,0)
	 if res1 == None: continue
	 idx1,sgn1 = res1
	 r = 2*i+1
   	 res2 = bits.action(idx1,r,0)
  	 if res2 == None: continue
	 idx2,sgn2 = res2
	 q = 2*i+1
         res3 = bits.action(idx2,q,1)
     	 if res3 == None: continue
     	 idx3,sgn3 = res3
	 p = 2*i
         res4 = bits.action(idx3,p,1)
       	 if res4 == None: continue
     	 idx4,sgn4 = res4
	 sigma[idx4] += U*sgn1*sgn2*sgn3*sgn4*vec[idx]
   return sigma
#==================================================================
