from fpeps2017.ptensor.include import np
from fpeps2017.ptensor.parray_helper import einsum 
from fpeps2017 import peps
from fpeps2017 import peps_h
import fpeps_util
import dumpVcoeff

# Settings
np.random.seed(5)
nr=2
nc=3
pdim=4
bond=2
auxbond=None
rfac=0.8

def test_save():
    print '\n[test_gen]'
    shape = (nr,nc)
    # simple peps
    peps0 = peps.random(shape, pdim, bond, fac=rfac)
    vec = peps.flatten(peps0)
    np.save('data/peps_vec2by3',vec)
    dumpVcoeff.dump(peps0)
    return 0

def test_load():
    print '\n[test_load]'
    shape = (nr,nc)
    # simple peps
    vec = np.load('data/peps_vec2by3.npy')
    peps0 = peps.aspeps(vec, (nr,nc), pdim, bond)
    ovlp,etot = energy_fpeps2by2(peps0)
    ovlp,php = nAverage_fpeps2by2(peps0)
    print 'nAveraged=',ovlp,php,php/ovlp
    exit()

    def energy_fn(vec, pdim, bond):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
        PP,PHP = energy_fpeps2by2(P)
	e = PHP/PP
        print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
        return PHP/PP
    def bound_energy_fn(vec):
        return energy_fn(vec, pdim, bond)

    vec = peps.flatten(peps0)
    import scipy.optimize
    import autograd
    
    deriv = autograd.grad(bound_energy_fn)
    print 'nparams=',len(vec)
    def save_vec(vec):
	fname = 'peps_vec1'
	np.save(fname,vec)
	print ' --- save vec into fname=',fname
	return 0
    # Optimize
    result = scipy.optimize.minimize(bound_energy_fn, jac=deriv, x0=vec,\
		    		     tol=1.e-4, callback=save_vec)
    P0 = peps.aspeps(result.x, (nr,nc), pdim, bond)
    print "final =",energy_fn(peps.flatten(P0), pdim, bond)
    return 0

def nAverage_fpeps2by2(peps0):
    debug = False	
    nr,nc = peps0.shape
    # <P|P>
    tmp = peps0[0,1].copy() 
    p23 = tmp.ptys[0]
    d23 = tmp.dims[0]
    p1 = tmp.ptys[2]
    d1 = tmp.dims[2]
    swap = fpeps_util.genSwap(p23,d23,p1,d1)
    peps1 = peps.copy(peps0)
    peps1[0,1] = einsum('pludr,pUuP->PlUdr',peps1[0,1],swap)
    ovlp = peps.dot(peps1,peps1)
    if debug:
       print
       print "Ovlp=",peps.dot(peps0,peps0)
       print "Ovlp=",ovlp
    nloc = fpeps_util.genNloc()
    # \hat{N} is a sum of local terms 
    nav = 0.0
    for i in range(nr):
       for j in range(nc):
          peps2 = peps.copy(peps1)
	  peps2[i,j] = einsum('Pp,pludr->Pludr',nloc,peps1[i,j])
          nav += peps.dot(peps1,peps2)
    return ovlp,nav
    
def energy_fpeps2by2(peps0):
    debug = True
    nr,nc = peps0.shape
    # <P|P>
    tmp = peps0[0,1].copy() 
    p23 = tmp.ptys[0]
    d23 = tmp.dims[0]
    p1 = tmp.ptys[1]
    d1 = tmp.dims[1]
    swap1 = fpeps_util.genSwap(p23,d23,p1,d1)
    # In fact, for 'uniform' distribution swap1=swap2
    tmp = peps0[0,2].copy() 
    p23 = tmp.ptys[0]
    d23 = tmp.dims[0]
    p1 = tmp.ptys[2]
    d1 = tmp.dims[2]
    swap2 = fpeps_util.genSwap(p23,d23,p1,d1)
    # 'Bosonic peps'
    peps1 = peps.copy(peps0)
    peps1[0,1] = einsum('pludr,pLlP->PLudr',peps1[0,1],swap1)
    peps1[0,2] = einsum('pludr,pUuP->PlUdr',peps1[0,2],swap2)
    ovlp = peps.dot(peps1,peps1)
    if debug:
       print
       print "Ovlp(bare)=",peps.dot(peps0,peps0)
       print "Ovlp(swap)=",ovlp

    etot = 0.0

    # <P|U[i]|P>    
    if debug: print "\nU-terms:"
    U = 1.5
    uterm = fpeps_util.genUterm(U)
    for j in range(nc):
       for i in range(nr):
	  peps2 = peps.copy(peps1)
	  peps2[i,j] = einsum('Pp,pludr->Pludr',uterm,peps1[i,j])
	  eloc = peps.dot(peps1,peps2)
	  if debug: print '(i,j)=',(i,j),' <U>=',eloc
	  etot = etot + eloc
    exit()

#    #
#    #   a b  c d
#    #   | |  | |
#    #   | 2--*-3
#    #   |/   |/
#    #   0----1
#    #
#    tC_aa,tA_aa = fpeps_util.genTaa()
#    tC_bb,tA_bb = fpeps_util.genTbb()
#    
#    # H02 = vertical bond
#    peps2 = peps1.copy()
#    peps2[0,0] = einsum('xqp,pludr->qlxudr',tC_aa,peps1[0,0]).merge_adjpair(2,3)
#    peps2[1,0] = einsum('xqp,pludr->qluxdr',tA_aa,peps1[1,0]).merge_adjpair(3,4)
#    elocA = peps.dot(peps1,peps2)
#    peps2 = peps1.copy()
#    peps2[0,0] = einsum('xqp,pludr->qlxudr',tC_bb,peps1[0,0]).merge_adjpair(2,3)
#    peps2[1,0] = einsum('xqp,pludr->qluxdr',tA_bb,peps1[1,0]).merge_adjpair(3,4)
#    elocB = peps.dot(peps1,peps2)
#    etot = etot + 2.0*(elocA+elocB)
#    if debug: 
#       print
#       print 't02aa=',elocA
#       print 't02bb=',elocB
#    
#    # H23 = horizontal bond
#    #     |    |
#    #     *----*
#    #     |/   |/
#    #   --2----3--
#    #    /    /
#    peps2 = peps1.copy()
#    peps2[1,0] = einsum('xqp,pludr->qludxr',tC_aa,peps1[1,0]).merge_adjpair(4,5)
#    peps2[1,1] = einsum('xqp,pludr->qxludr',tA_aa,peps1[1,1]).merge_adjpair(1,2)
#    elocA = peps.dot(peps1,peps2)
#    peps2 = peps1.copy()
#    peps2[1,0] = einsum('xqp,pludr->qludxr',tC_bb,peps1[1,0]).merge_adjpair(4,5)
#    peps2[1,1] = einsum('xqp,pludr->qxludr',tA_bb,peps1[1,1]).merge_adjpair(1,2)
#    elocB = peps.dot(peps1,peps2)
#    etot = etot + 2.0*(elocA+elocB)
#    if debug:
#       print
#       print 't23aa=',elocA
#       print 't23bb=',elocB
#   
#    # parity along physical index
#    parity = fpeps_util.genParitySgn([0,1],[2,2])
#
#    # H01 - hbond
#    peps2 = peps1.copy()
#    peps2[0,0] = einsum('xqp,pludr->qludxr',tC_aa,peps1[0,0]).merge_adjpair(4,5)
#    peps2[0,1] = einsum('xqp,pludr->qxludr',tA_aa,peps1[0,1]).merge_adjpair(1,2)
#    peps2[1,0] = einsum( 'qp,pludr->qludr',parity,peps1[1,0])
#    peps2[1,1] = einsum( 'qp,pludr->qludr',parity,peps1[1,1])
#    elocA = peps.dot(peps1,peps2)
#    peps2 = peps1.copy()
#    peps2[0,0] = einsum('xqp,pludr->qludxr',tC_bb,peps1[0,0]).merge_adjpair(4,5)
#    peps2[0,1] = einsum('xqp,pludr->qxludr',tA_bb,peps1[0,1]).merge_adjpair(1,2)
#    peps2[1,0] = einsum( 'qp,pludr->qludr',parity,peps1[1,0])
#    peps2[1,1] = einsum( 'qp,pludr->qludr',parity,peps1[1,1])
#    elocB = peps.dot(peps1,peps2)
#    etot = etot + 2.0*(elocA+elocB)
#    if debug:
#       print
#       print 't01aa=',elocA
#       print 't01bb=',elocB
# 
#    # H13 - vbond
#    peps2 = peps1.copy()
#    peps2[0,1] = einsum( 'qp,pludr->qludr',parity,peps1[0,1])
#    peps2[0,1] = einsum('xqp,pludr->qlxudr',tC_aa,peps2[0,1]).merge_adjpair(2,3)
#    peps2[1,1] = einsum('xqp,pludr->qluxdr',tA_aa,peps1[1,1]).merge_adjpair(3,4)
#    peps2[1,1] = einsum( 'qp,pludr->qludr',parity,peps2[1,1])
#    elocA = peps.dot(peps1,peps2)
#    peps2 = peps1.copy()
#    peps2[0,1] = einsum( 'qp,pludr->qludr',parity,peps1[0,1])
#    peps2[0,1] = einsum('xqp,pludr->qlxudr',tC_bb,peps2[0,1]).merge_adjpair(2,3)
#    peps2[1,1] = einsum('xqp,pludr->qluxdr',tA_bb,peps1[1,1]).merge_adjpair(3,4)
#    peps2[1,1] = einsum( 'qp,pludr->qludr',parity,peps2[1,1])
#    elocB = peps.dot(peps1,peps2)
#    etot = etot + 2.0*(elocA+elocB)
#    if debug:
#       print
#       print 't13aa=',elocA
#       print 't13bb=',elocB
    return ovlp,etot

#==============================================================================

if __name__ == '__main__':
   test_save()
   test_load()
