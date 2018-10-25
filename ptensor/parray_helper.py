from include import np,npeinsum
from parray import PArray
import re
import null
import copy

def einsum(idx_str, *tensors):
    """
    Support 
    1. basic: ('ij,jk->ik',a,b)
    2. full contract: ('ij,ij',a,b)
    3. tproduct ('ij,kl->ijkl',a,b)
    4. multiple ('ijkl,jkl,mn->imn',a,b,c)
    """
    # Split '->'
    ifext = idx_str.count('->') # external indices
    if ifext: 
       idx0, idx1 = idx_str.split('->')
    else:
       idx0 = idx_str
       idx1 = None

    # Recursively split into binary einsum
    if idx0.count(',') > 1:
        indices  = re.split(',',idx0)
	# first contract two with maximal no. of shared indices
        n_shared_max = 0
        for i in range(len(indices)):
            for j in range(i):
                tmp = list(set(indices[i]).intersection(indices[j]))
                n_shared_indices = len(tmp)
                if n_shared_indices > n_shared_max:
                    n_shared_max = n_shared_indices
                    shared_indices = tmp
                    [a,b] = [i,j]
        tensors = list(tensors)
        A, B = tensors[a], tensors[b]
        idxA, idxB = indices[a], indices[b]
        idx_out = list(idxA+idxB)
	idx_out = [x for x in idx_out if x not in shared_indices]
	if len(idx_out) == 0:
           print ' error: contraction like "ijkl,ijkl,m" not allowed!'
	   exit(1)
	idx_out = "".join(idx_out)
	C = einsum(idxA+","+idxB+"->"+idx_out, A, B)
	indices.pop(a) # Remove the item at the given position in the list
        indices.pop(b)
        indices.append(idx_out)
        tensors.pop(a)
        tensors.pop(b)
        tensors.append(C)
	if idx1 == None:
	   return einsum(",".join(indices),*tensors)
	else:	   
   	   return einsum(",".join(indices)+"->"+idx1,*tensors)

    # Binary einsum case: C = contract(A,B)
    A, B = tensors
    # A&B
    idxA, idxB = idx0.split(',')
    # A
    idxA = list(idxA)
    rkA  = len(idxA) 
    dicA = dict(zip(idxA,range(rkA)))
    # B
    idxB = list(idxB)
    rkB  = len(idxB)
    dicB = dict(zip(idxB,range(rkB)))
    if idx1 == None:
       # Fully contracted case
       for a in idxA:
	  assert a in idxB
       # Such case einsum('ijkl,ijmn',p1,p2) is not supported!
       # output indices need to be explicitly specified.
       C = null.NULL
       assert A.ptys == B.ptys
       for ix in np.ndindex(A.shape):
          assert A.ifpc(ix) == B.ifpc(ix)
          if not A.ifpc(ix): continue 
          C += npeinsum(idx0,A[ix],B[ix])
    else:
       # Partial contraction
       # 1. Define shape of C via external indices
       idxC = list(idx1)
       rkC  = len(idxC)
       # get ptys & dims
       ptysC = []
       dimsC = []
       for ic in idxC:
	  if ic in idxA:
	     ptysC.append(A.ptys[dicA[ic]])
	     dimsC.append(A.dims[dicA[ic]])
	  else:
	     ptysC.append(B.ptys[dicB[ic]])
	     dimsC.append(B.dims[dicB[ic]])
       C = PArray(ptysC,dimsC)
       # 2. Contract via internal indices
       idxAB = list(set(idxA).intersection(idxB))
       rkAB  = len(idxAB)
       idxAC = [a for a in idxA if a not in idxAB]
       idxBC = [b for b in idxB if b not in idxAB]
       ictrA = [dicA[a] for a in idxAB]
       ectrA = [dicA[a] for a in idxAC]
       ictrB = [dicB[b] for b in idxAB]
       ectrB = [dicB[b] for b in idxBC]
       # Permutation
       idxCC = idxAC+idxBC
       dicC  = dict(zip(idxCC,range(rkC)))
       ordC  = [dicC[ic] for ic in idxC]  
       for ixA in np.ndindex(A.shape):
	  if not A.ifpc(ixA): continue
	  ecA = [ixA[ia] for ia in ectrA] 
	  icA = [ixA[ia] for ia in ictrA]
	  ipA = [A.ptys[ia][ic] for ia,ic in zip(ictrA,icA)] 
          for ixB in np.ndindex(B.shape):
	     if not B.ifpc(ixB): continue
	     ecB = [ixB[ib] for ib in ectrB]
	     icB = [ixB[ib] for ib in ictrB]
	     ipB = [B.ptys[ib][ic] for ib,ic in zip(ictrB,icB)] 
	     # Construct contracted internal position
	     if icA != icB: continue
	     assert ipA == ipB
	     # Construcut ixC; 
	     # there is no need to check ptys conversation,
	     # also no need to compute shape in advance. 
	     ecC = ecA + ecB
	     ixC = tuple([ecC[ic] for ic in ordC])
	     # It does not support += yet?
	     C[ixC] = C[ixC] + npeinsum(idx_str,A[ixA],B[ixB])
    return C

# add two tensors along one direction
def concatenate(t1,t2,axis):
   ptys = copy.deepcopy(t1.ptys)
   dims = copy.deepcopy(t2.dims)
   ptys.pop(axis)
   dims.pop(axis)
   ptys.insert(axis,t1.ptys[axis]+t2.ptys[axis])
   dims.insert(axis,t1.dims[axis]+t2.dims[axis])
   new = PArray(ptys,dims)
   nqi = len(t1.ptys[axis])
   for ix in np.ndindex(new.shape): 
      if not new.ifpc(ix): continue
      if ix[axis] < nqi:
	 new[ix] = t1[ix]
      else:
	 nix = list(ix)
	 nix.pop(axis)
	 nix.insert(axis,ix[axis]-nqi)
	 nix = tuple(nix)
	 new[ix] = t2[nix]
   new = new.merge_axis(axis)
   return new


if __name__ == '__main__':

    testCases = []#1,2,3,4,5,7]
    for icase in testCases:
       print 'icase=',icase
       if icase == 1:
          # 1. basic: ('ij,jk->ik',a,b)
          p1 = PArray([[0],[0,1],[1]],[[1],[3,3],[2]])
          p2 = PArray([[0,1],[0,1]],[[3,3],[4,4]])
          pf = einsum('ijl,jk->kil',p1,p2)
       elif icase == 2:
          # 2. full contract: ('ij,ij',a,b)
          p1 = PArray([[0],[0,1]],[[1],[3,4]])
	  p1[0,0][:] = 1. 
	  pf = einsum('ij,ij',p1,p1)
       elif icase == 3:
          # 3. tproduct ('ij,kl->ijkl',a,b)
          p1 = PArray([[0],[0,1]],[[4],[4,4]])
          p2 = PArray([[0,1],[0,1]],[[4,4],[4,4]])
          pf = einsum('ij,kl->ijkl',p1,p2)
       elif icase == 4:
          # 4. multiple ('ijkl,jkl,mn->imn',a,b,c)
          p1 = PArray([[0],[0,1],[0,1],[0,1]],[[4],[4,4],[4,4],[4,4]])
          p2 = PArray([[0,1],[0,1],[0,1]],[[4,4],[4,4],[4,4]])
          p3 = PArray([[0],[0,1]],[[4],[4,4]])
          pf = einsum('ijkl,jkl,mn->imn',p1,p2,p3)
       elif icase == 5:
          # 5. multiple ('ijkl,jkl,i',a,b,c)
          p1 = PArray([[0],[0,1],[0,1],[0,1]],[[4],[4,4],[4,4],[4,4]])
          p2 = PArray([[0,1],[0,1]],[[4,4],[4,4]])
          p3 = PArray([[0],[0,1]],[[4],[4,4]])
          pf = einsum('ikjl,kl,ij',p1,p2,p3)
       elif icase == 6:
          # 6. multiple ('ijkl,ijkl,m',a,b,c)
          p1 = PArray([[0],[0,1],[0,1],[0,1]],[[4],[4,4],[4,4],[4,4]])
          p2 = PArray([[0],[0,1],[0,1],[1]],[[4],[4,4],[4,4],[4]])
          p3 = PArray([[0]],[[4]])
          pf = einsum('ijkl,ijkl,m',p1,p2,p3)
       elif icase == 7:
          p1 = PArray([[0],[0,1],[0,1],[0,1]],[[4],[4,4],[4,4],[4,4]])
	  p2 = p1
	  pf = einsum('ijkl,ijmn',p1,p2)
       print ' final result =',pf

    import autograd
    # 2. full contract: ('ij,ij',a,b)
    ptys = [[0],[0,1],[1]]
    dims = [[2],[3,4],[5]]
    p1 = PArray(ptys,dims)
    p1.random()
    vec = p1.ravel()
    
    def fun(vec):
       p2 = PArray(ptys,dims)
       p2.fill(vec)
       pf = einsum('ijk,ijk',p2,p2)
       return pf
    deriv = autograd.grad(fun)
    print fun(vec)
    print deriv(vec)
    
    def fun2(vec):
       p2 = PArray(ptys,dims)
       p2.fill(vec)
       pf = einsum('ijl,kjl->ik',p2,p2)
       return pf[0,0][0,0]
    deriv2 = autograd.grad(fun2)
    print fun2(vec)
    print deriv2(vec)
    
    print 'concatenate'
    p1 = PArray([[0],[0,1],[0,1]],[[1],[3,3],[2,2]])
    p1.random()
    p2 = PArray([[0],[0,1],[0]],[[1],[3,3],[2]])
    p2.random()
    p1.prt()
    p2.prt()
    p3 = concatenate(p1,p2,2)
    p3.prt()
