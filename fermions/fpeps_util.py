from fpeps2017.ptensor.include import np
from fpeps2017.ptensor.parray_helper import einsum 
from fpeps2017 import peps
from fpeps2017 import peps_h
from fpeps2017.ptensor.parray import PArray

# exp(i*theta*N)
def genRmat(theta):
    rmat = PArray([[0,1],[0,1]],[[2,2],[2,2]])
    rmat[0,0][0,0] = 1.
    rmat[0,0][1,1] = np.exp(1.j*theta*2)
    rmat[1,1][0,0] = np.exp(1.j*theta)
    rmat[1,1][1,1] = np.exp(1.j*theta)
    return rmat

# SWAP(ludr) = delta[l,r]*delta[u,r]*S[p(l),p(u)]
# S[p(l),p(u)] = -1 if p(l)=p(u)=1
def genSwap(p23,d23,p1,d1,fac=-1.0):
    swap = PArray([p23,p1,p1,p23],[d23,d1,d1,d23])
    for il,ptyl in enumerate(p23):
       idenl = np.identity(d23[il])
       for iu,ptyu in enumerate(p1):
          idenu = np.identity(d1[iu])
          if (ptyl,ptyu) == (1,1):
             swap[il,iu,iu,il] = np.einsum('lr,ud->ludr',idenl,idenu)*fac
          else:
 	     swap[il,iu,iu,il] = np.einsum('lr,ud->ludr',idenl,idenu)
    return swap

# ---P---
def genParitySgn(p,d):
    pt = PArray([p,p],[d,d])
    for idx,ip in enumerate(p):
       if ip == 0:
	  pt[idx,idx] = np.identity(d[idx])
       elif ip == 1:
	  pt[idx,idx] = -np.identity(d[idx])
    return pt

def genNloc():
    nloc = PArray([[0,1],[0,1]],[[2,2],[2,2]])
    nloc[0,0][1,1] = 2.
    nloc[1,1][0,0] = 1.
    nloc[1,1][1,1] = 1.
    return nloc

def genUterm(U):
    ut = PArray([[0,1],[0,1]],[[2,2],[2,2]])
    ut[0,0][1,1] = U
    return ut

def genTaa():
    # cre[alpha]-ann[alpha]  (up-down)
    tC_aa = PArray([[1],[0,1],[0,1]],[[1],[2,2],[2,2]])
    tC_aa[0,1,0][0,0,0] = 1.0 # <u|A+|0>
    tC_aa[0,0,1][0,1,1] = 1.0 # <ud|A+|d>
    tA_aa = PArray([[1],[0,1],[0,1]],[[1],[2,2],[2,2]])
    tA_aa[0,0,1][0,0,0] = -1.0 # <0|A|u> (-t)
    tA_aa[0,1,0][0,1,1] = -1.0 # <d|A|ud>(-t)
    return tC_aa,tA_aa

def genTbb():
    # cre[beta]-ann[beta]  (up-down)
    tC_bb = PArray([[1],[0,1],[0,1]],[[1],[2,2],[2,2]])
    tC_bb[0,1,0][0,1,0] =  1.0 # <d|B+|0>
    tC_bb[0,0,1][0,1,0] = -1.0 # <ud|B+|u>
    tA_bb = PArray([[1],[0,1],[0,1]],[[1],[2,2],[2,2]])
    tA_bb[0,0,1][0,0,1] = -1.0 # <0|B|d> (-t)
    tA_bb[0,1,0][0,0,1] =  1.0 # <u|B|ud>(-t)
    return tC_bb,tA_bb

# Fermionic contraction
def fdot(pepsa,pepsb,auxbond=None):
    epeps0 = fepeps(pepsa,pepsb)
    return peps.contract_cpeps(epeps0, auxbond)

def fepeps(pepsa,pepsb):
    shape = pepsa.shape
    epeps = np.empty(shape, dtype=np.object)
    for i in range(shape[0]):
        for j in range(shape[1]):
	    p0 = pepsa[i,j].ptys[1]
	    d0 = pepsa[i,j].dims[1]
	    p1 = pepsb[i,j].ptys[2]
	    d1 = pepsb[i,j].dims[2]
            swapLU = genSwap(p0,d0,p1,d1)
	    p0 = pepsb[i,j].ptys[4]
	    d0 = pepsb[i,j].dims[4]
	    p1 = pepsa[i,j].ptys[3]
	    d1 = pepsa[i,j].dims[3]
            swapRD = genSwap(p0,d0,p1,d1)
            epeps[i,j] = einsum("lUYX,pXuWr,pLYDV,VWdR->lLuUdDrR",\
			         swapLU,pepsa[i,j],pepsb[i,j],swapRD)
	    epeps[i,j] = epeps[i,j].merge([[0,1],[2,3],[4,5],[6,7]])
    return epeps
