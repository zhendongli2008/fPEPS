#========================
# Basic operation of MPO
#  * MPO*MPS
#========================
from ptensor.parray import PArray
from ptensor.include import np
from ptensor.parray_helper import einsum

def mapply(mpo,mps):
    """
    apply mpo to mps, or apply mpo to mpo
    """
    nsites = len(mpo)
    assert len(mps)==nsites
    ret = [None]*nsites
    if len(mps[0].shape)==3: 
        # mpo x mps
        for i in xrange(nsites):
	    mt = einsum("apqb,cqd->acpbd",mpo[i],mps[i])
            ret[i] = mt.merge([[0,1],[2],[3,4]])
    elif len(mps[0].shape)==4: 
        # mpo x mpo
        for i in xrange(nsites):
            mt = einsum("apqb,cqrd->acprbd",mpo[i],mps[i])
            ret[i] = mt.merge([[0,1],[2],[3],[4,5]])
    return ret
