import bits
from ptensor.include import np
import scipy.sparse as sps

def fromConf2Bits(conf):
    nr,nc = conf.shape
    x = 0
    idx = 0
    for i in range(nr):
       for j in range(nc):
          if conf[i,j] == 1: x = bits.setBit(x,idx)
          idx += 1
    return x

def fromBits2Conf(bt,nr,nc):
    conf = np.zeros([nr,nc], dtype=np.int)
    idx = 0
    for i in range(nr):
       for j in range(nc):
          if bits.testBit(bt,idx): conf[i,j] = 1
          idx += 1
    return conf

#
# Heisenberg Hamiltonian
#
SzMat = np.array([[0.5,0.],[0.,-0.5]])
SpMat = np.array([[0. ,1.],[0.,0.  ]])
SmMat = np.array([[0. ,0.],[1.,0.  ]])

def genHeisenbergHmn(m,n,mtx):
    diff = m^n # xor: 0,1->1; 1,0->1
    ndiff = bits.bitCount(diff)
    Hmn = 0.
    if ndiff == 0:
       for i in range(mtx.shape[0]):
	  for j in mtx.rows[i]:
	     # (a,b) or (b,a)
	     if bits.testBit(m,i)^bits.testBit(m,j): 
		Hmn -= 0.25
	     # (a,a) or (b,b)
	     else:
	        Hmn += 0.25
    elif ndiff == 1:
       # speciality of Heisenberg Hamiltonian - double flip 
       Hmn = 0. 
    elif ndiff == 2:
       i,j = bits.toOrblst(diff)
       if j in mtx.rows[i]:
          mi = bits.testBit(m,i) 
	  mj = bits.testBit(m,j) 
	  ni = bits.testBit(n,i)
	  nj = bits.testBit(n,j)
	  Hmn = SzMat[mi,mj]*SzMat[ni,nj] \
	      + 0.5*SpMat[mi,mj]*SmMat[ni,nj] \
	      + 0.5*SmMat[mi,mj]*SpMat[ni,nj]
    return Hmn

def genHeisenbergHlst(nr,nc):
    # ordering
    def orbs(i,j):
       return i*nc+j
    row = []
    col = []
    # Hbond
    for i in range(nr):
       for j in range(nc-1):
	   row.append(orbs(i,j))
	   col.append(orbs(i,j+1))
    # Vbond
    for i in range(nr-1):
       for j in range(nc):
	   row.append(orbs(i,j))
	   col.append(orbs(i+1,j))
    # Linked list
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.lil_matrix.html
    mdim = max(max(row),max(col))+1
    mtx = sps.lil_matrix((mdim,mdim))
    mtx[row,col] = 1
    return mtx

def genHeisenbergMove(x,nr,nc):
    nspin = nr*nc
    ups = [i for i in range(nspin) if not bits.testBit(x,i)]
    dws = [i for i in range(nspin) if bits.testBit(x,i)]
    i = np.random.choice(ups)
    j = np.random.choice(dws)
    y = bits.toggleBit(x,i)
    y = bits.toggleBit(y,j)
    return y

def genHeisenbergConnected(x,nr,nc):
    # ordering
    def orbs(i,j):
       return i*nc+j
    conf = fromBits2Conf(x,nr,nc)
    clst = [x]
    # Hbond: -
    for i in range(nr):
       for j in range(nc-1):
	   if conf[i,j]^conf[i,j+1]:
	      y = bits.toggleBit(x,orbs(i,j))
	      y = bits.toggleBit(y,orbs(i,j+1))
	      clst.append(y)
    # Vbond: |
    for i in range(nr-1):
       for j in range(nc):
	   if conf[i,j]^conf[i+1,j]: 
	      y = bits.toggleBit(x,orbs(i,j))
	      y = bits.toggleBit(y,orbs(i+1,j))
	      clst.append(y)
    return clst

if __name__ == '__main__':
    x = 23130
    y = 23098
    print bin(x)
    print bin(y)
    nr = 4
    nc = 4
    mtx = genHeisenbergHlst(nr,nc)
    print mtx
    print genHeisenbergHmn(x,x,mtx)
    print genHeisenbergHmn(x,y,mtx)
    print genHeisenbergHmn(y,y,mtx)

    print bin(y)
    conf = fromBits2Conf(y,nr,nc)
    print conf[0]
    print conf
    # 0b101101000111010
    # [[0 1 0 1]  -  row0
    #  [1 1 0 0]
    #  [0 1 0 1]
    #  [1 0 1 0]]

    y = genHeisenbergMove(x,nr,nc)
    print
    print fromBits2Conf(y,nr,nc)
    print genHeisenbergConnected(y,nr,nc)
