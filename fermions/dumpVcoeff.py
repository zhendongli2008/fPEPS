from fpeps2017.ptensor.include import np
from fpeps2017.ptensor.parray_helper import einsum 
import fpeps_util

# Global addresing for physical indices
addr = np.zeros((2,2),dtype=np.int)
addr[0,0] = 0 # |0>
addr[0,1] = 3 # |ud>
addr[1,0] = 1 # |d>
addr[1,1] = 2 # |u>

def dump(peps0):
    shape = peps0.shape
    if shape == (2,2):
       vcoeff = dump2by2(peps0)
       np.save('data/vcoeff2by2',vcoeff)
    elif shape == (2,3):
       vcoeff = dump2by3(peps0)
       np.save('data/vcoeff2by3',vcoeff)
    elif shape == (3,3):
       vcoeff = dump3by3(peps0)
       np.save('data/vcoeff3by3',vcoeff)
    print '\n[dumpVcoeff.dump] shape=',shape
    print 'shape=',vcoeff.shape
    print 'vcoeff<P|P>=',vcoeff.dot(vcoeff)
    print '<P|P>=',vcoeff.dot(vcoeff)
    return 0

def dump2by2(peps0):
    p1 = peps0[0,1].ptys[0]
    d1 = peps0[0,1].dims[0]
    p23 = peps0[1,0].ptys[4]
    d23 = peps0[1,0].dims[4]
    swap = fpeps_util.genSwap(p23,d23,p1,d1)

    coeff = einsum('amEpJ,IJHqk,bnrEF,FcIG,dGsHl->abcdnmpqrslk',\
	    	    peps0[0,0],peps0[0,1],peps0[1,0],swap,peps0[1,1])
      	 			       # 0123456789|10|11
    coeff = coeff.merge([[0],[1],[2],[3],\
		    	 [4,5],[6,7],[8,9],[10,11]])
    coeff.prt()
    print 'einsum<PEPS|PEPS>=',einsum('abcdNPRL,abcdNPRL',coeff,coeff)
  
    nr,nc = peps0.shape
    nsite = nr*nc
    vtensor = np.zeros([4]*nsite)
    for qa in range(2):
       for qb in range(2):
          for qc in range(2):
             for qd in range(2):
		if coeff[qa,qb,qc,qd,0,0,0,0].shape == 0: continue
		c0 = coeff[qa,qb,qc,qd,0,0,0,0].reshape(2,2,2,2)
		for ia in range(2):
		   for ib in range(2):
		      for ic in range(2):
		         for id in range(2):
			    xa = addr[qa,ia] 
			    xb = addr[qb,ib]
			    xc = addr[qc,ic]
			    xd = addr[qd,id]
			    vtensor[xa,xb,xc,xd] = c0[ia,ib,ic,id]
    vcoeff = vtensor.flatten()
    return vcoeff

def dump2by3(peps0):
    p1 = peps0[0,1].ptys[0]
    d1 = peps0[0,1].dims[0]
    p23 = peps0[1,0].ptys[4]
    d23 = peps0[1,0].dims[4]
    swap = fpeps_util.genSwap(p23,d23,p1,d1)

    coeff = einsum('amEpJ,IJHqK,bnrEF,FcIG,dGsHL,LeOM,OKNuw,fMtNv->abcdefnmpqurstvw',\
	    	    peps0[0,0],peps0[0,1],peps0[1,0],swap,peps0[1,1],\
		    swap,peps0[0,2],peps0[1,2])
    # abcdef|nmpqurstvw => Only four auxilliary indices left !!!
    coeff = coeff.merge([[0],[1],[2],[3],[4],[5],\
		         [6,7],[8,9,10],[11,12,13],[14,15]])
    coeff.prt()
    # 4^6 = 4096
    print 'einsum<PEPS|PEPS>=',einsum('abcdefMNOP,abcdefMNOP',coeff,coeff)
 
    nr,nc = peps0.shape
    nsite = nr*nc
    vtensor = np.zeros([4]*nsite)
    for qa in range(2):
     for qb in range(2):
      for qc in range(2):
       for qd in range(2):
        for qe in range(2):
         for qf in range(2):
            if coeff[qa,qb,qc,qd,qe,qf,0,0,0,0].shape == 0: continue
	    c0 = coeff[qa,qb,qc,qd,qe,qf,0,0,0,0].reshape([2]*nsite)
            for ia in range(2):
	     for ib in range(2):
	      for ic in range(2):
	       for id in range(2):
	        for ie in range(2):
	         for if0 in range(2):
	            xa = addr[qa,ia] 
		    xb = addr[qb,ib]
		    xc = addr[qc,ic]
		    xd = addr[qd,id]
		    xe = addr[qe,ie]
		    xf = addr[qf,if0]
		    vtensor[xa,xb,xc,xd,xe,xf] = c0[ia,ib,ic,id,ie,if0]
    vcoeff = vtensor.flatten()
    return vcoeff

def dump3by3(peps0):
    p1 = peps0[0,1].ptys[0]
    d1 = peps0[0,1].dims[0]
    p23 = peps0[1,0].ptys[4]
    d23 = peps0[1,0].dims[4]
    swap = fpeps_util.genSwap(p23,d23,p1,d1)
   
    coeff = einsum('alAsC,bkBAD,DGIH,LHMJO,ICJtN,cjpBE,EdGF,FeLK,fKqMP,PgRQ,QhWV,iVrXm,ORTS,WSXUn,TNUuo->abcdefghijklpqrstumno',\
         	    peps0[0,0],peps0[1,0],swap,peps0[1,1],peps0[0,1],\
     	            peps0[2,0],swap,swap,peps0[2,1],\
     	            swap,swap,peps0[2,2],swap,peps0[1,2],peps0[0,2])
    # abcdef|nmpqurstvw => Only four auxilliary indices left !!!
    coeff = coeff.merge([[0],[1],[2],[3],[4],[5],[6],[7],[8],\
        	         [9,10,11],[12,13,14],[15,16,17],[18,19,20]])
    coeff.prt()
    # 4^9 
    print 'einsum<PEPS|PEPS>=',einsum('abcdefghiMNOP,abcdefghiMNOP',coeff,coeff)
  
    nr,nc = peps0.shape
    nsite = nr*nc
    vtensor = np.zeros([4]*nsite)
    for qa in range(2):
     for qb in range(2):
      for qc in range(2):
       for qd in range(2):
        for qe in range(2):
         for qf in range(2):
          for qg in range(2):
           for qh in range(2):
            for qi in range(2):
             if coeff[qa,qb,qc,qd,qe,qf,qg,qh,qi,0,0,0,0].shape == 0: continue
             c0 = coeff[qa,qb,qc,qd,qe,qf,qg,qh,qi,0,0,0,0].reshape([2]*nsite)
             for ia in range(2):
              for ib in range(2):
               for ic in range(2):
                for id in range(2):
                 for ie in range(2):
                  for if0 in range(2):
                   for ig in range(2):
                    for ih in range(2):
                     for ii in range(2):
                       xa = addr[qa,ia] 
        	       xb = addr[qb,ib]
        	       xc = addr[qc,ic]
        	       xd = addr[qd,id]
        	       xe = addr[qe,ie]
        	       xf = addr[qf,if0]
        	       xg = addr[qg,ig]
        	       xh = addr[qh,ih]
        	       xi = addr[qi,ii]
        	       vtensor[xa,xb,xc,xd,xe,xf,xg,xh,xi] = \
        	   	   c0[ia,ib,ic,id,ie,if0,ig,ih,ii]
    vcoeff = vtensor.flatten()
    return vcoeff
