#=========================================================
#
# Convension of PEPS tensor: 
#  * pludr
#
# Basic functions:
#  * Initialize PEPS by zeros/random/given configurations
#  * Add two PEPS - require merge of quantum blocks !!!
#  * Contract PEPS
#
#=========================================================
from ptensor.parray import PArray
from ptensor.include import np,dataType
from ptensor.parray_helper import einsum
import mps
import mpo

# Return physical dimension of a tensor: 2 or 4
def get_pdim(peps):
    return sum(peps[0,0].dims[0])

# Return bond dimension: pludr
def get_bond(peps):
    return peps[0,0].dims[2][0]

# In this way, we always set the global parity 
# of PEPS to even, since the boundary are even!
def set_auxparity(auxdim):
   if len(auxdim) == 1: # boundary
      parity = [0]
   else:
      parity = [0,1]
   return parity

#
# (vertical)
#  | | | |		   NOT matrix notation      
# -*-*-*-*- (horizontal)   .                        
#  | | | |		   .
# -*-*-*-*-		   .	
#  | | | |		   (2,0) (2,1) (2,2)
# -*-*-*-*-		   (1,0) (1,1) (1,2)
#  | | | |		   (0,0) (0,1) (0,2) . . .
#
def zeros(shape,pdim,bond):
    debug = False	
    if pdim == 2:
       ppty = [0,1]
       pdim = [1,1]
    elif pdim == 4:
       ppty = [0,1]
       pdim = [2,2] # {|0>,|ud>}, {|d>,|u>}
    else:
       print 'error: not supported for PEPS with pdim=',pdim
       exit(1)
    # initialization
    nr,nc = shape
    peps = np.empty(shape, dtype=np.object)
    # dimension of bonds, ludr
    hor_dims = [[1]]+[[bond]*2]*(nc-1)+[[1]]
    ver_dims = [[1]]+[[bond]*2]*(nr-1)+[[1]]
    for i in range(nr):
        for j in range(nc):
            ldim = hor_dims[j]
	    rdim = hor_dims[j+1]
	    udim = ver_dims[i+1]
	    ddim = ver_dims[i]
	    lpty = set_auxparity(ldim)
	    rpty = set_auxparity(rdim)
	    upty = set_auxparity(udim)
	    dpty = set_auxparity(ddim)
	    ptys = [ppty,lpty,upty,dpty,rpty]
	    dims = [pdim,ldim,udim,ddim,rdim]
	    peps[i,j] = PArray(ptys,dims)
	    if debug: print (i,j),ptys,dims,peps[i,j].prt()
    return peps

# Initialized by random numbers
def random(shape,pdim,bond,fac=1.0):
    peps = zeros(shape,pdim,bond)
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            peps[i,j].random(fac)
    return peps

# Add two PEPS: assuming a unique bond dimension for each irrep!!!
def add(pepsa,pepsb):
    assert pepsa.shape == pepsb.shape # two-by-two
    shape = pepsa.shape
    pdim  = get_pdim(pepsa)
    bonda = pepsa[0,0].dims[-1][0]
    bondb = pepsb[0,0].dims[-1][0]
    bond  = bonda+bondb
    pepsc = zeros(shape,pdim,bond)
    for i in range(shape[0]):
        for j in range(shape[1]):
	    # LOOP OVER NON-ZERO BLOCKS
	    for ix in np.ndindex(pepsc[i,j].shape):
	        if pepsc[i,j].ifpc(ix):
	  	    n,la,ua,da,ra = pepsa[i,j][ix].shape
	  	    n,lc,uc,dc,rc = pepsc[i,j][ix].shape
		    pepsc[i,j][ix][:,:la,:ua,:da,:ra] = pepsa[i,j][ix]
		    # pepsb
		    l1,l2 = la,lc
		    u1,u2 = ua,uc
		    d1,d2 = da,dc
		    r1,r2 = ra,rc
		    # Boundary case
		    if i == 0: # last row 
		       assert da == dc == 1
		       d1,d2 = 0,1
	      	    elif i == shape[0]-1: # last row
		       assert ua == uc == 1
		       u1,u2 = 0,1
		    if j == 0: # first col
		       assert la == lc == 1
		       l1,l2 = 0,1
		    if j == shape[1]-1:
		       assert ra == rc == 1
		       r1,r2 = 0,1
		    pepsc[i,j][ix][:,l1:l2,u1:u2,d1:d2,r1:r2] = pepsb[i,j][ix]
    return pepsc

def copy(pepsa):
    shape = pepsa.shape
    pepsb = np.empty(shape, dtype=np.object)
    for i in range(shape[0]):
       for j in range(shape[1]):
           pepsb[i,j] = pepsa[i,j].copy()
    return pepsb

#=======
# confs
#=======
def create(pdim,config):
    shape = config.shape
    peps0 = zeros(shape,pdim,1)
    # bottom row
    pu = [None]*shape[1]
    for j in range(shape[1]):
        pu[j] = config[0,j]  
        peps0[0,j][config[0,j],0,pu[j],0,0] = np.ones(1).reshape(1,1,1,1,1)
    # upper row
    for i in range(1,shape[0]-1):
        pd = [q for q in pu]
        for j in range(shape[1]):
            pu[j] = pd[j]^config[i,j]
            peps0[i,j][config[i,j],0,pu[j],pd[j],0] = np.ones(1).reshape(1,1,1,1,1)
    # top row
    i = shape[0]-1
    pl = 0
    for j in range(shape[1]):
       pr = pl^(pu[j]^config[i,j])
       peps0[i,j][config[i,j],pl,0,pu[j],pr] = np.ones(1).reshape(1,1,1,1,1)
       pl = pr
    return peps0

# Take an inefficient way to circurvment the need for changing
# the parity of site tensor after specifying a index 0 or 1.
# Note: This would be a problem, if create allows to create peps with
# different bond dimension for different parity chanels, such that
# the total bond dimension for aux indices is exactly 1.
def ceval(peps,config,auxbond):
    pdim = get_pdim(peps)
    peps0 = create(pdim,config)
    return dot(peps0,peps,auxbond)

#==============
# DOT two PEPS
#==============
def dot(pepsa,pepsb,auxbond=None):
    epeps0 = epeps(pepsa,pepsb)
    return contract_cpeps(epeps0, auxbond)

def epeps(pepsa,pepsb):
    shape = pepsa.shape
    epeps = np.empty(shape, dtype=np.object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            epeps[i,j]=einsum("pludr,pLUDR->lLuUdDrR",pepsa[i,j],pepsb[i,j])
	    epeps[i,j] = epeps[i,j].merge([[0,1],[2,3],[4,5],[6,7]])
    return epeps
 
#=======================================================
# Contract PEPS via nrows MPOs act on MPS + Compression
#=======================================================
def get_boundaryMPS(cpeps,key):
    nr,nc = cpeps.shape
    if key == 'd': j = 0
    if key == 'u': j = nr-1
    cmps0 = [None]*nc
    for i in range(nc):
	cmps0[i] = cpeps[j,i].merge([[0],[1,2],[3]])
    return cmps0

def get_rowMPO(cpeps,i):
    nr,nc = cpeps.shape
    cmpo = [None]*nc
    for j in range(nc):
        cmpo[j] = cpeps[i,j].copy()
    return cmpo

def contract_cpeps(cpeps,auxbond):
    cmps0 = get_boundaryMPS(cpeps,'d')
    for i in range(1,cpeps.shape[0]-1):
        cmpo = get_rowMPO(cpeps,i)
        cmps0 = mpo.mapply(cmpo,cmps0)
	if auxbond is not None: # compress
            cmps0 = mps.compress(cmps0,auxbond)
    cmps1 = get_boundaryMPS(cpeps,'u')
    return mps.dot(cmps0,cmps1)
 
#=====================================
# vector rep. packed for optimization
#=====================================
def size(peps):
    size=0
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            size += peps[i,j].size()
    return size
    
def flatten(peps):
    vec=np.empty((0),dtype=dataType)
    for i in range(peps.shape[0]):
        for j in range(peps.shape[1]):
            vec = np.append(vec,peps[i,j].ravel())
    return vec

# Convert vec to PEPS structure
def aspeps(vec,shape,pdim,bond):
    peps0 = zeros(shape,pdim,bond)
    assert vec.size == size(peps0)
    ptr = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            nelem = peps0[i,j].size()
            peps0[i,j].fill(vec[ptr:ptr+nelem])
            ptr += nelem
    return peps0

# Normalized
def add_noise(peps,fac=1.0):
    vec = flatten(peps)
    vec = vec + fac*np.random.uniform(-1,1,vec.shape)
    #vec = vec/np.linalg.norm(vec)
    peps_new = aspeps(vec,peps.shape,\
		      get_pdim(peps),get_bond(peps))
    return peps_new
