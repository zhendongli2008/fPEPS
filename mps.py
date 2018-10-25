#========================
# Basic operation of MPS
#  * dot
#  * compression
#========================
from ptensor.parray import PArray
from ptensor.include import np
from ptensor.parray_helper import einsum
import autograd

def dot(mpsa,mpsb):
    """
    dot product of two mps
    """
    assert len(mpsa)==len(mpsb)
    nsites = len(mpsa)
    e0 = PArray([[0],[0]],[[1],[1]])
    e0[0,0] = np.ones(1).reshape(1,1) 
    for i in xrange(nsites):
        tmp = einsum('ij,jnr->inr',e0,mpsb[i])
	e0 = einsum('inl,inr->lr',mpsa[i],tmp)
    return e0[0,0][0,0]

#=================
# SVD compression
#=================
def get_bdims(mps):
    bdims = [mps[i].dims[2] for i in range(len(mps)-1)]
    return bdims

def compress(mps,trunc=1.e-12,debug=False):
    f0,mps_new = compress_side(mps,"l",trunc,debug)
    f1,mps_new = compress_side(mps_new,"r",trunc,debug)
    f2,mps_new = compress_side(mps_new,"l",trunc,debug)
    if debug: print ' summary of fidelity=',(f0,f1,f2),f0*f1*f2
    return mps_new

def compress_side(mps,side,trunc=1.e-12,debug=False):
    """
    inp: canonicalise MPS (or MPO)

    trunc=0: just canonicalise
    0<trunc<1: sigma threshold
    trunc>1: number of renormalised vectors to keep
   
    returns:
         truncated or canonicalised MPS
    """
    if debug: print '\n[mps.compress_side] with side=',side
    assert side in ["l","r"]
    if side == "l":
       fidelity,mps_new = compress_left(mps,trunc,debug)
    else:
       mps_new = flip(mps)
       fidelity,mps_new = compress_left(mps_new,trunc,debug)
       mps_new = flip(mps_new)
    return fidelity,mps_new

def flip(mps):
    nsites = len(mps)
    mps_new = [None]*nsites
    for isite in range(nsites):
        mps_new[nsites-isite-1] = mps[isite].transpose([2,1,0])
    return mps_new

def compress_left(mps,trunc=1.e-12,debug=False):
    nrm2 = dot(mps, mps)
    nsites = len(mps)
    ret_mps = [None]*nsites
    # Note: in EPEPS, the physical dim can be O(D^2); 
    #       auxbond O(D^4) after MPO apply to MPS. 
    res0 = mps[0] 
    for i in xrange(1,nsites):
	# Merge (l,n,r)
	res = res0.merge_adjpair(0,1)
        u0,sigma0,vt0 = autograd.numpy.linalg.svd(res[0,0],full_matrices=False)
	u1,sigma1,vt1 = autograd.numpy.linalg.svd(res[1,1],full_matrices=False)
        assert len(sigma0) == len(sigma1)
        if trunc==0:
            m_trunc=len(sigma0)
        elif trunc<1.:
            # count how many sing vals < trunc    
	    total = np.sqrt(np.sum(sigma0**2)+np.sum(sigma1**2))
            normed_sigma0=sigma0/total
            normed_sigma1=sigma1/total
            m_trunc=max(len([s for s in normed_sigma0 if s >trunc]),\
			len([s for s in normed_sigma1 if s >trunc]))
        else:
            m_trunc=int(trunc)
            m_trunc=min(m_trunc,len(sigma0))
	if debug:
           s0r = np.sum(sigma0[0:m_trunc]**2)
	   s0t = np.sum(sigma0**2)
           s1r = np.sum(sigma1[0:m_trunc]**2)
	   s1t = np.sum(sigma1**2)
	   print ' i=',i,' len(sig0)=',len(sigma0),' m_trunc=',m_trunc,' trunc=',trunc
	   print ' sigma0^2[tot,rem,dis] =',(s0t,s0r,s0t-s0r)
	   print ' sigma1^2[tot,rem,dis] =',(s1t,s1r,s1t-s1r)
	u0=u0[:,0:m_trunc]
        sigma0=np.diag(sigma0[0:m_trunc])
        vt0=vt0[0:m_trunc,:]
        u1=u1[:,0:m_trunc]
        sigma1=np.diag(sigma1[0:m_trunc])
        vt1=vt1[0:m_trunc,:]
	# Update site
	ptys = res0.ptys
	dims = res0.dims
	dimr = [m_trunc,m_trunc]
	site = PArray([ptys[0],ptys[1],[0,1]],[dims[0],dims[1],dimr])
	if len(ptys[0]) == 1:
	   site[0,0,0] = u0.reshape(site[0,0,0].shape) # parity-0
	   site[0,1,1] = u1.reshape(site[0,1,1].shape) # parity-1
  	else:
	   nblk00 = dims[0][0]*dims[1][0]
	   nblk01 = dims[0][0]*dims[1][1]
	   site[0,0,0] = u0[:nblk00,:].reshape(site[0,0,0].shape) # parity-0 
	   site[1,1,0] = u0[nblk00:,:].reshape(site[1,1,0].shape) # parity-0
	   site[0,1,1] = u1[:nblk01,:].reshape(site[0,1,1].shape) # parity-1 
	   site[1,0,1] = u1[nblk01:,:].reshape(site[1,1,0].shape) # parity-1
	ret_mps[i-1] = site
	# Assignment of vt
	vt = PArray([[0,1],[0,1]],[dimr,dims[2]])
	vt[0,0] = np.dot(sigma0,vt0)
	vt[1,1] = np.dot(sigma1,vt1)
	res0 = einsum('ij,jnr->inr',vt,mps[i])
    # Last one
    ret_mps[nsites-1] = res0
    # Check
    fidelity = dot(ret_mps, mps)/(nrm2+1.e-10) # zero-EPES
    if debug: print ' nrm2=',nrm2,' fidelity=',fidelity
    return fidelity,ret_mps
