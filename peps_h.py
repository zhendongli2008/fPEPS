from ptensor.parray import PArray
from ptensor.include import np
from ptensor.parray_helper import einsum,concatenate
import peps

# Conventions:
# alpha - 0
# beta  - 1
# then Sp = [[0,1]
# 	     [0,0]]

# Sz = .5*np.array([[1.,0.],[0.,-1.]])
def get_Sz():
    Sz = PArray([[0,1],[0,1]],[[1,1],[1,1]])
    Sz[0,0] = np.array(0.5).reshape(1,1)
    Sz[1,1] = np.array(-0.5).reshape(1,1)
    return Sz

# Sp = np.array([[0.,1.],[0.,0.]])
def get_Sp():
    Sp = PArray([[1],[0,1],[0,1]],[[1],[1,1],[1,1]]) # To match with PEPS qnums!
    Sp[0,0,1] = np.array(1).reshape(1,1,1)
    return Sp

# Sm = np.array([[0.,0.],[1.,0.]])
def get_Sm():
    Sm = PArray([[1],[0,1],[0,1]],[[1],[1,1],[1,1]])
    Sm[0,1,0] = np.array(1).reshape(1,1,1)
    return Sm

Sz = get_Sz()
Sp = get_Sp()
Sm = get_Sm()

# Evaluate: vec{S}_i*vec{S}_j = Sz_i*Sz_j + 0.5*(Sp_i*Sm_j+c.c.) 
#
# Horizontal:
#
#  parity=odd,bond=1
#    |----|
#    |/   |/
# ===*====*===
#   /    /
#  [i,j] [i,j+1]
#
def eval_hbond(pepsa0, pepsa, i, j, auxbond):
    debug = False
    # Sz*Sz
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("pq,qludr->pludr",Sz,pepsa[i,j])
    pepsb[i,j+1] = einsum("pq,qludr->pludr",Sz,pepsa[i,j+1])
    valzz = peps.dot(pepsa0,pepsb,auxbond)
    if debug: print (i,j),'valzz=',valzz
    # Sp*Sm
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("xpq,qludr->pludxr",Sp,pepsa[i,j]  ).merge_adjpair(4,5)
    pepsb[i,j+1] = einsum("xpq,qludr->pxludr",Sm,pepsa[i,j+1]).merge_adjpair(1,2)
    valpm = peps.dot(pepsa0,pepsb,auxbond)
    if debug: print (i,j),'valpm=',valpm
    # Sm*Sp
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("xpq,qludr->pludxr",Sm,pepsa[i,j]  ).merge_adjpair(4,5)
    pepsb[i,j+1] = einsum("xpq,qludr->pxludr",Sp,pepsa[i,j+1]).merge_adjpair(1,2)
    valmp = peps.dot(pepsa0,pepsb,auxbond)
    if debug: print (i,j),'valmp=',valmp
    return valzz + 0.5*(valpm+valmp)

# Vertial 
#   |
#   * [i+1,j]
#   |
#   * [i]
#   |
def eval_vbond(pepsa0, pepsa, i, j,auxbond):
    debug = False
    # Sz*Sz
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("pq,qludr->pludr",Sz,pepsa[i,j])
    pepsb[i+1,j] = einsum("pq,qludr->pludr",Sz,pepsa[i+1,j])
    valzz = peps.dot(pepsa0,pepsb,auxbond)
    if debug: print (i,j),'valzz=',valzz
    # Sp*Sm
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("xpq,qludr->plxudr",Sp,pepsa[i,j]  ).merge_adjpair(2,3)
    pepsb[i+1,j] = einsum("xpq,qludr->pluxdr",Sm,pepsa[i+1,j]).merge_adjpair(3,4)
    valpm = peps.dot(pepsa0,pepsb,auxbond)
    if debug: print (i,j),'valpm=',valpm
    # Sm*Sp
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("xpq,qludr->plxudr",Sm,pepsa[i,j]  ).merge_adjpair(2,3)
    pepsb[i+1,j] = einsum("xpq,qludr->pluxdr",Sp,pepsa[i+1,j]).merge_adjpair(3,4)
    valmp = peps.dot(pepsa0,pepsb,auxbond)
    if debug: print (i,j),'valmp=',valmp
    return valzz + 0.5*(valpm+valmp)

# Evaluate <PEPSa|Si*Sj|PEPSb> for each term of Si*Sj
def eval_heish(pepsa, pepsb, auxbond, debug=False):
    shape = pepsa.shape
    nr,nc = shape
    val = 0.
    for i in range(nr):
        for j in range(nc-1):
            val += eval_hbond(pepsa,pepsb,i,j,auxbond)
	    if debug: print '(i,j)/val=',(i,j),val
    for i in range(nr-1):
        for j in range(nc):
            val += eval_vbond(pepsa,pepsb,i,j,auxbond)
	    if debug: print '(i,j)/val=',(i,j),val
    return val

#   # A special prod (1+xSi*Sj)
#   def product(pepsa,auxbond,x):
#       pepsb = peps.copy(pepsa)
#       nr,nc = pepsa.shape
#   
#       Sz = PArray([[0,1],[0,1]],[[1,1],[1,1]])
#       Sz[0,0] = np.sqrt(x)*np.array(0.5).reshape(1,1)
#       Sz[1,1] = np.sqrt(x)*np.array(-0.5).reshape(1,1)
#       Sp = PArray([[1],[0,1],[0,1]],[[1],[1,1],[1,1]]) # To match with PEPS qnums!
#       Sp[0,0,1] = np.sqrt(x)*np.array(1).reshape(1,1,1)
#       Sp = Sp.merge([[0,1],[2]])
#       Sm = PArray([[1],[0,1],[0,1]],[[1],[1,1],[1,1]])
#       Sm[0,1,0] = np.sqrt(x)*np.array(1).reshape(1,1,1)
#       Sm = Sm.merge([[0,1],[2]])
#   
#       # Hbond: left-right
#       for i in range(nr):
#          for j in range(nc-1):
#   	  tmp1 = pepsb[i,j].copy()
#   	  tmp2 = einsum("pq,qludr->pludr",Sz, pepsb[i,j])
#   	  tmp1 = concatenate(tmp1,tmp2,4)
#   	  tmp2 = einsum("Pq,qludr->Pludr",Sp, pepsb[i,j])
#   	  tmp1 = concatenate(tmp1,tmp2,4)
#   	  tmp2 = einsum("Pq,qludr->Pludr",Sm, pepsb[i,j])
#   	  tmp1 = concatenate(tmp1,tmp2,4)
#   	  pepsb[i,j] = tmp1.copy()
#             tmp1 = pepsb[i,j+1].copy()
#   	  tmp2 = einsum("pq,qludr->pludr",Sz, pepsb[i,j+1])
#   	  tmp1 = concatenate(tmp1,tmp2,1)
#   	  tmp2 = einsum("Pq,qludr->Pludr",Sm, pepsb[i,j+1])
#   	  tmp1 = concatenate(tmp1,tmp2,1)
#   	  tmp2 = einsum("Pq,qludr->Pludr",Sp, pepsb[i,j+1])
#   	  tmp1 = concatenate(tmp1,tmp2,1)
#   	  pepsb[i,j+1] = tmp1.copy()
#       # Vbond: up-down
#       for i in range(nr-1):
#          for j in range(nc):
#   	  tmp1 = pepsb[i,j].copy()
#   	  tmp2 = einsum("pq,qludr->pludr",Sz, pepsb[i,j])
#   	  tmp1 = concatenate(tmp1,tmp2,2)
#   	  tmp2 = einsum("Pq,qludr->Pludr",Sp, pepsb[i,j])
#   	  tmp1 = concatenate(tmp1,tmp2,2)
#   	  tmp2 = einsum("Pq,qludr->Pludr",Sm, pepsb[i,j])
#   	  tmp1 = concatenate(tmp1,tmp2,2)
#   	  pepsb[i,j] = tmp1.copy()
#   	  # right
#             tmp1 = pepsb[i+1,j].copy()
#   	  tmp2 = einsum("pq,qludr->pludr",Sz, pepsb[i+1,j])
#   	  tmp1 = concatenate(tmp1,tmp2,3)                
#   	  tmp2 = einsum("Pq,qludr->Pludr",Sm, pepsb[i+1,j])
#   	  tmp1 = concatenate(tmp1,tmp2,3)                
#   	  tmp2 = einsum("Pq,qludr->Pludr",Sp, pepsb[i+1,j])
#   	  tmp1 = concatenate(tmp1,tmp2,3)
#   	  pepsb[i+1,j] = tmp1.copy()
#       val = peps.dot(pepsb,pepsa,auxbond)
#       return val 
