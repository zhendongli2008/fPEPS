from fpeps2017.ptensor.include import np
from fpeps2017.ptensor.parray_helper import einsum 
from fpeps2017 import peps
from fpeps2017 import peps_h
import fpeps_util
import dumpVcoeff

# Settings
np.random.seed(5)
nr=2
nc=2
pdim=4
bond=3
auxbond=None
U=1.5
rfac=0.8
debug=True

def test_save():
    print '[test_gen]'
    shape = (nr,nc)
    # simple peps
    peps0 = peps.random(shape, pdim, bond, fac=rfac)
    vec = peps.flatten(peps0)
    np.save('data/peps_vec2by2',vec)
    dumpVcoeff.dump(peps0)
    return 0

def test_load():
    print '[test_load]'
    shape = (nr,nc)
    # simple peps
    vec = np.load('data/peps_vec2by2.npy')
    peps0 = peps.aspeps(vec, (nr,nc), pdim, bond)
    ovlp,etot = energy_fpeps2by2(peps0)
    ovlp,php = nAverage_fpeps2by2(peps0)
    print 'etot=',ovlp,etot,etot/ovlp
    print 'nAverage=',ovlp,php,php/ovlp
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
    elst = []
    def save_vec(vec):
	fname = 'data/peps_vec1'
	np.save(fname,vec)
	e = bound_energy_fn(vec)
	elst.append(e)
	print ' --- save vec into fname=',fname,' e=',e
	return 0
    # Optimize
    result = scipy.optimize.minimize(bound_energy_fn, jac=deriv, x0=vec,\
		    		     tol=1.e-4, callback=save_vec)
    P0 = peps.aspeps(result.x, (nr,nc), pdim, bond)
    print "final =",energy_fn(peps.flatten(P0), pdim, bond)
    np.save('data/energy',elst)
    return 0

def nAverage_fpeps2by2(peps0):
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
    
    etot = 0.0

    # <P|U[i]|P>    
    if debug: print "\nU-terms:"
    uterm = fpeps_util.genUterm(U)
    for i in range(nr):
       for j in range(nc):
	  peps2 = peps.copy(peps1)
	  peps2[i,j] = einsum('Pp,pludr->Pludr',uterm,peps1[i,j])
	  eloc = peps.dot(peps1,peps2)
	  if debug: print '(i,j)=',(i,j),' <U>=',eloc
	  etot = etot + eloc

    #
    #   a b  c d
    #   | |  | |
    #   | 1--*-3
    #   |/   |/
    #   0----2
    #
    tC_aa,tA_aa = fpeps_util.genTaa()
    tC_bb,tA_bb = fpeps_util.genTbb()
    
    # Test local terms: 
    if debug:
       tCA_aa = einsum('xqr,xrp->qp',tC_aa,tA_aa)
       tCA_bb = einsum('xqr,xrp->qp',tC_bb,tA_bb)
       for i in [0,1]:
          peps2 = peps.copy(peps1)
          peps2[i,0] = einsum('qp,pludr->qludr',tCA_aa,peps1[i,0])
          elocA = 2.0*peps.dot(peps1,peps2)
          print 't['+str(i)+']aa=',elocA
          peps2 = peps.copy(peps1)
          peps2[i,0] = einsum('qp,pludr->qludr',tCA_bb,peps1[i,0])
          elocA = 2.0*peps.dot(peps1,peps2)
          print 't['+str(i)+']bb=',elocA
 
    # Jordan-Wigner sign factor for i\=j 
    parity = fpeps_util.genParitySgn([0,1],[2,2])
    tC_aa = einsum('xqp,pr->xqr',tC_aa,parity)
    tC_bb = einsum('xqp,pr->xqr',tC_bb,parity)

    # H01 = vertical bond
    peps2 = peps.copy(peps1)
    peps2[0,0] = einsum('xqp,pludr->qlxudr',tC_aa,peps1[0,0]).merge_adjpair(2,3)
    peps2[1,0] = einsum('xqp,pludr->qluxdr',tA_aa,peps1[1,0]).merge_adjpair(3,4)
    elocA = 2.0*peps.dot(peps1,peps2)

    peps2 = peps.copy(peps1)
    peps2[0,0] = einsum('xqp,pludr->qlxudr',tC_bb,peps1[0,0]).merge_adjpair(2,3)
    peps2[1,0] = einsum('xqp,pludr->qluxdr',tA_bb,peps1[1,0]).merge_adjpair(3,4)
    elocB = 2.0*peps.dot(peps1,peps2)
    etot = etot + (elocA+elocB)
    if debug: 
       print
       print 't01aa=',elocA
       print 't01bb=',elocB

    # H13 = horizontal bond
    #     |    |
    #     *----*
    #     |/   |/
    #   --1----3--
    #    /    /
    peps2 = peps.copy(peps1)
    peps2[1,0] = einsum('xqp,pludr->qludxr',tC_aa,peps1[1,0]).merge_adjpair(4,5)
    peps2[1,1] = einsum('xqp,pludr->qxludr',tA_aa,peps1[1,1]).merge_adjpair(1,2)
    elocA = 2.0*peps.dot(peps1,peps2)
    peps2 = peps.copy(peps1)
    peps2[1,0] = einsum('xqp,pludr->qludxr',tC_bb,peps1[1,0]).merge_adjpair(4,5)
    peps2[1,1] = einsum('xqp,pludr->qxludr',tA_bb,peps1[1,1]).merge_adjpair(1,2)
    elocB = 2.0*peps.dot(peps1,peps2)
    etot = etot + (elocA+elocB)
    if debug:
       print
       print 't13aa=',elocA
       print 't13bb=',elocB
   
    # H02 - hbond
    peps2 = peps.copy(peps1)
    peps2[0,0] = einsum('xqp,pludr->qludxr',tC_aa,peps1[0,0]).merge_adjpair(4,5)
    peps2[0,1] = einsum('xqp,pludr->qxludr',tA_aa,peps1[0,1]).merge_adjpair(1,2)
    peps2[1,0] = einsum( 'qp,pludr->qludr',parity,peps1[1,0])
    peps2[1,1] = einsum( 'qp,pludr->qludr',parity,peps1[1,1])
    elocA = 2.0*peps.dot(peps1,peps2)
    peps2 = peps.copy(peps1)
    peps2[0,0] = einsum('xqp,pludr->qludxr',tC_bb,peps1[0,0]).merge_adjpair(4,5)
    peps2[0,1] = einsum('xqp,pludr->qxludr',tA_bb,peps1[0,1]).merge_adjpair(1,2)
    peps2[1,0] = einsum( 'qp,pludr->qludr',parity,peps1[1,0])
    peps2[1,1] = einsum( 'qp,pludr->qludr',parity,peps1[1,1])
    elocB = 2.0*peps.dot(peps1,peps2)
    etot = etot + (elocA+elocB)
    if debug:
       print
       print 't02aa=',elocA
       print 't02bb=',elocB
 
    # H23 - vbond
    peps2 = peps.copy(peps1)
    peps2[0,1] = einsum( 'qp,pludr->qludr',parity,peps1[0,1])
    peps2[0,1] = einsum('xqp,pludr->qlxudr',tC_aa,peps2[0,1]).merge_adjpair(2,3)
    peps2[1,1] = einsum('xqp,pludr->qluxdr',tA_aa,peps1[1,1]).merge_adjpair(3,4)
    peps2[1,1] = einsum( 'qp,pludr->qludr',parity,peps2[1,1])
    elocA = 2.0*peps.dot(peps1,peps2)
    peps2 = peps.copy(peps1)
    peps2[0,1] = einsum( 'qp,pludr->qludr',parity,peps1[0,1])
    peps2[0,1] = einsum('xqp,pludr->qlxudr',tC_bb,peps2[0,1]).merge_adjpair(2,3)
    peps2[1,1] = einsum('xqp,pludr->qluxdr',tA_bb,peps1[1,1]).merge_adjpair(3,4)
    peps2[1,1] = einsum( 'qp,pludr->qludr',parity,peps2[1,1])
    elocB = 2.0*peps.dot(peps1,peps2)
    etot = etot + (elocA+elocB)
    if debug:
       print
       print 't23aa=',elocA
       print 't23bb=',elocB
       print
       print 'ovlp,etot=',ovlp,etot
    return ovlp,etot

#==============================================================================

def test_plot():
    elst = np.load('data/energy.npy')
    import matplotlib.pyplot as plt
    efci = -3.700331728271
    plt.plot(np.log10(elst-efci),'ro-')
    plt.savefig('data/convergence.pdf')
    plt.show()
    return 0

if __name__ == '__main__':
   test_save()
   test_load()
   exit()
   test_plot()
