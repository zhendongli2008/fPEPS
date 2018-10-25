from include import np,dataType
import copy
import null

#
# Parity conserving block-sparse tensor
#
class PArray(np.ndarray):
   #
   # parity - array of parities for each dim [[0],[1],[0,1]] 
   # dimens - array of dimfor each dim [[3],[4],[5,5]] 
   #
   def __new__(cls,ptys,dims):
       shape = map(lambda x:len(x),ptys)
       shape1 = map(lambda x:len(x),dims)
       assert shape == shape1
       obj = np.ndarray.__new__(cls,shape,dtype=np.object)
       obj[:] = null.Null()
       obj.ptys = copy.deepcopy(ptys)
       obj.dims = copy.deepcopy(dims)
       # Initialize
       for ix in np.ndindex(obj.shape): # np.ndindex(3, 2, 1)
	  if not obj.ifpc(ix): continue
	  shp = obj.get_shp(ix)
	  obj[ix] = np.zeros(shp,dtype=dataType)
       return obj

   def __array_finalize(self, obj):
       if obj is None and obj.dtype != object:
           raise RuntimeError

   # shape of a given block
   def get_shp(self,ix):
       return [dim[i] for i,dim in zip(ix,self.dims)]
  
   def prt(self):
       print 'PArray.prt'
       print ' ptys=',self.ptys
       print ' dims=',self.dims
       nrm2 = 0.0
       for ix in np.ndindex(self.shape): 
	  line = None
	  if self.ifpc(ix): 
	     nrm2i = np.linalg.norm(self[ix])**2
	     nrm2 += nrm2i
	     line = ' norm2='+str(nrm2i)
          print ' iblk=',ix,self.ifpc(ix),self[ix].shape,line
       print ' norm2=',nrm2,'\n'
       return 0

   def scale(self,fac=1.0):
       for ix in np.ndindex(self.shape): 
	  if self.ifpc(ix):
	     self[ix] = fac*self[ix]
       return 0

   # Generate random entries
   def random(self,fac=1.0):
       for ix in np.ndindex(self.shape): 
	  if self.ifpc(ix):
	     self[ix] = fac*np.random.uniform(-1,1,self[ix].shape)
       return 0

   def size(self):
       size = 0
       for ix in np.ndindex(self.shape): 
	  if self.ifpc(ix): size += np.product(self[ix].shape)
       return size

   def ravel(self):
       vec = np.empty((0))
       for ix in np.ndindex(self.shape): 
	  if self.ifpc(ix):
	     vec = np.append(vec,np.ravel(self[ix]))
       return vec

   def fill(self,vec):
       ptr = 0
       for ix in np.ndindex(self.shape): 
	  if self.ifpc(ix):
	     shape = self[ix].shape
	     nelem = np.product(shape)
	     self[ix] = vec[ptr:ptr+nelem].reshape(shape)
	     ptr += nelem
       return vec

   # Parities
   def get_px(self,ix):
       px = [pt[i] for pt,i in zip(self.ptys,ix)]
       return px 

   def ifpc(self,ix):
       return ifpc(self.get_px(ix))

   # TRANSPOSE & MERGE
   def transpose(self,axes):
       ptys_new = [self.ptys[i] for i in axes]
       dims_new = [self.dims[i] for i in axes] 
       new = PArray(ptys_new,dims_new)
       # OLD_INDEX
       for ix in np.ndindex(self.shape):
	   if self.ifpc(ix):
	       ix_new = tuple([ix[i] for i in axes])
               new[ix_new] = self[ix].transpose(axes)
       return new

   # merge quantum numbers along one axis 
   def merge_axis(self,i):
       ptys = copy.deepcopy(self.ptys)
       dims = copy.deepcopy(self.dims)
       dic = {}
       for idx,ip in enumerate(ptys[i]):
	  if ip not in dic:
	     dic[ip] = [idx]
	  else:
	     dic[ip].append(idx)
       qsec = sorted(dic.keys())
       nq = len(qsec)
       dsec = [sum([dims[i][iq] for iq in dic[qsec[idx]]]) for idx in range(nq)]
       ptys.pop(i)
       ptys.insert(i,qsec)
       dims.pop(i)
       dims.insert(i,dsec)
       new = PArray(ptys,dims)
       for ix in np.ndindex(new.shape): 
	  if not new.ifpc(ix): continue
	  tensors = []
	  for j in dic[qsec[ix[i]]]:
	     px = list(ix)
	     px[i] = j
	     tensors.append(self[tuple(px)])
	  new[ix] = np.concatenate(tensors,axis=i)
       return new

   def merge_adjpair(self,i0,i1):
       assert i0 + 1 == i1
       p0 = self.ptys[i0]
       p1 = self.ptys[i1]
       d0 = self.dims[i0]
       d1 = self.dims[i1]
       p01 = []
       d01 = []
       ns0 = len(p0)
       ns1 = len(p1)
       for j0 in range(ns0):
	  for j1 in range(ns1):
	     p01.append(p0[j0]^p1[j1])
	     d01.append(d0[j0]*d1[j1])
       ptys = copy.deepcopy(self.ptys)
       ptys.pop(i0)
       ptys.pop(i0)
       ptys.insert(i0,p01)
       dims = copy.deepcopy(self.dims)
       dims.pop(i0)
       dims.pop(i0)
       dims.insert(i0,d01)
       new = PArray(ptys,dims)
       for ix in np.ndindex(self.shape): 
	  if not self.ifpc(ix): continue
	  nix = np.ravel_multi_index([ix[i0],ix[i1]],(ns0,ns1)) 
	  nix = ix[:i0]+tuple([nix])+ix[i1+1:]
	  shape = list(self[ix].shape)
	  nshape = shape[i0]*shape[i1]
	  shape.pop(i0)
	  shape.pop(i0)
	  shape.insert(i0,nshape)
	  new[nix] = self[ix].reshape(shape)
       new = new.merge_axis(i0)
       return new

   # 'deepcopy'
   def copy(self):
       return self.merge_axis(0)

   def merge(self,partition):
       new = self.copy()
       # A simple trick to avoid 
       for ipart in partition[-1::-1]:
	  m = len(ipart)
	  if m == 1: continue
	  for j in range(m-2,-1,-1):
	     new = new.merge_adjpair(ipart[j],ipart[j+1])
       return new

# Check if a list of parities return even
# 0 - even; 1 - odd; parity rule (xor): 
# 0^0=0, 0^1=1, 1^0=1, 1^1=0
def ifpc(x):
    ix = list(x)
    ifp = 0^ix[0]
    for i in ix[1:]:
       ifp = ifp^i
    return not ifp

if __name__ == '__main__':
    ix1 = [0,0,0,0]
    ix2 = [0,0,1,0]
    ix3 = [0,1,1,0]
    ix4 = [0]
    ix5 = [1]
    print ifpc(ix1)
    print ifpc(ix2)
    print ifpc(ix3)
    print ifpc(ix4)
    print ifpc(ix5)
    
    p1 = PArray([[0],[0,1],[1]],[[1],[3,3],[2]])
    p1.prt()
    p2 = p1.transpose([0,2,1])
    p2.prt()
      
    p3 = p2
    p3.prt()
    
    p3 = PArray([[0],[0,1],[0,1],[1]],[[1],[3,3],[2,5],[3]])
    p3.random()
    p3.prt()
    p4 = p3.merge_adjpair(1,2)
    p4.prt()
    
    print 'p3'
    p3.prt()
    p4 = p3.merge([[0],[1,2,3]])
    p4.prt()
