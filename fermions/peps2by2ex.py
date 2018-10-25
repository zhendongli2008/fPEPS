from fpeps2017.ptensor.include import np
from fpeps2017.ptensor.parray_helper import einsum 
from fpeps2017 import peps
from fpeps2017 import peps_h
import fpeps_util
import dumpVcoeff

def reflection(peps0):
   nr,nc = peps0.shape
   peps = np.empty((nr,nc), dtype=np.object)
   for i in range(nr):	
      for j in range(nc):
         peps[i,j] = peps0[nr-i-1,j].transpose([0,1,3,2,4]) # pludr->pldur
   return peps     
	
# Settings
np.random.seed(5)
nr=2
nc=2
pdim=4
bond=3
auxbond=None
rfac=0.8
U=1.5
debug=False

def test_save():
    print '\n[test_gen]'
    shape = (nr,nc)
    # simple peps
    peps0 = peps.random(shape, pdim, bond, fac=rfac)
    vec = peps.flatten(peps0)
    np.save('data/peps_vec3by3',vec)
    dumpVcoeff.dump(peps0)
    return 0

def test_load():
    print '\n[test_load]'
    shape = (nr,nc)
    # simple peps
    vec = np.load('data/peps_vec3by3.npy')
    peps0 = peps.aspeps(vec, (nr,nc), pdim, bond)
    ovlp,php = nAverage_fpeps3by3(peps0)
    print 'nAveraged=',ovlp,php,php/ovlp
    ovlp,etot = energy_fpeps3by3(peps0,0.0)
    print 'ovlp,etot=',ovlp,etot,etot/ovlp

    ne = 4.0
    npts = 5
    xts = np.array([(2.0*i-1.0)/(2*npts)*np.pi for i in range(1,npts+1)])
    wts = np.array([np.pi/npts]*npts)*[np.exp(-1.j*x*ne) for x in xts]
    print '\nQuadrature[Chebyshev-Gauss]'
    print 'xts=',xts
    print 'wts=',wts

    def energy_fn2(vec, pdim, bond):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
	PP_tot = 0.
	PHP_tot = 0.
	for ipt in range(npts):
           PP,PHP = energy_fpeps3by3(P,xts[ipt])
	   PP_tot += PP*wts[ipt]
	   PHP_tot += PHP*wts[ipt]
	   #print ' PHP,PP',PHP,PP
	e = PHP_tot/PP_tot
	print ' PHP_tot,PP_tot=',PHP_tot,PP_tot
	print ' e_tot=',e
        return e

    def energy_fn(vec, pdim, bond):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
        PP,PHP = energy_fpeps3by3(P)
	e = PHP/PP
        print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
        return PHP/PP

    def bound_energy_fn(vec):
        return energy_fn2(vec, pdim, bond)

    vec = peps.flatten(peps0)
    import scipy.optimize
    import autograd
    
    deriv = autograd.grad(bound_energy_fn)
    print 'nparams=',len(vec)
    def save_vec(vec):
	fname = 'peps_vec1_3by3'
	np.save(fname,vec)
	print ' --- save vec into fname=',fname
	return 0
    # Optimize
    result = scipy.optimize.minimize(bound_energy_fn, jac=deriv, x0=vec,\
		    		     tol=1.e-3, callback=save_vec)
    P0 = peps.aspeps(result.x, (nr,nc), pdim, bond)
    print "final =",bound_energy_fn(peps.flatten(P0))
    return 0

def localRotation(peps0,theta):
    rmat = fpeps_util.genRmat(theta)
    nr,nc = peps0.shape
    peps1 = peps.copy(peps0)
    for i in range(nr):
       for j in range(nc):
	   peps1[i,j] = einsum('qp,pludr->qludr',rmat,peps1[i,j])
    return peps1

def nAverage_fpeps3by3(peps0):
    nr,nc = peps0.shape
    # <P|P>
    peps0b = reflection(peps0)
    peps1b = peps.copy(peps0b)
    for i in range(nr):
       for j in range(nc):
	  tmp = peps1b[i,j].copy()
          p0 = tmp.ptys[1]
          d0 = tmp.dims[1]
          p1 = tmp.ptys[2]
          d1 = tmp.dims[2]
          swapLU = fpeps_util.genSwap(p0,d0,p1,d1)
	  tmp = einsum('luUL,pLUdr->pludr',swapLU,tmp)
	  p0 = tmp.ptys[4]
          d0 = tmp.dims[4]
          p1 = tmp.ptys[3]
          d1 = tmp.dims[3]
          swapRD = fpeps_util.genSwap(p0,d0,p1,d1)
	  tmp = einsum('RDdr,pluDR->pludr',swapRD,tmp)
          peps1b[i,j] = tmp.copy()
    ovlp = fpeps_util.fdot(peps0b,peps1b)
    if debug:
       print
       print "Ovlp=",peps.dot(peps0,peps0)
       print "Ovlp=",ovlp

    nloc = fpeps_util.genNloc()
    # \hat{N} is a sum of local terms 
    nav = 0.0
    for j in range(nc):
       for i in range(nr):
          # Measure each local occupation <Ni> 
	  peps2b = peps.copy(peps1b)
	  peps2b[i,j] = einsum('Pp,pludr->Pludr',nloc,peps1b[i,j])
	  nii = fpeps_util.fdot(peps0b,peps2b)
          nav += nii 
	  print '(i,j)=',(i,j),'<Nloc>=',nii,nii/ovlp
    print 'nav=',ovlp,nav,nav/ovlp
    return ovlp,nav
    
def energy_fpeps3by3(peps0,theta):
    nr,nc = peps0.shape
    # <P|P>
    peps0b = reflection(peps0)
    
    # Acting local rotations: R(x)|P>
    peps0x = localRotation(peps0b,theta) 
    peps1b = peps.copy(peps0x)
    
    for i in range(nr):
       for j in range(nc):
	  tmp = peps1b[i,j].copy()
          p0 = tmp.ptys[1]
          d0 = tmp.dims[1]
          p1 = tmp.ptys[2]
          d1 = tmp.dims[2]
          swapLU = fpeps_util.genSwap(p0,d0,p1,d1)
	  tmp = einsum('luUL,pLUdr->pludr',swapLU,tmp)
          p0 = tmp.ptys[4]
          d0 = tmp.dims[4]
          p1 = tmp.ptys[3]
          d1 = tmp.dims[3]
          swapRD = fpeps_util.genSwap(p0,d0,p1,d1)
	  tmp = einsum('RDdr,pluDR->pludr',swapRD,tmp)
          peps1b[i,j] = tmp.copy()
    ovlp = fpeps_util.fdot(peps0b,peps1b)
    if debug:
       print
       print "Ovlp=",peps.dot(peps0,peps0)
       print "Ovlp=",ovlp

    etot = 0.0

    # <P|U[i]|P>    
    if debug: print "\nU-terms:"
    uterm = fpeps_util.genUterm(U)
    for j in range(nc):
       for i in range(nr):
	  peps2 = peps.copy(peps0)
	  peps2[i,j] = einsum('Pp,pludr->Pludr',uterm,peps0[i,j])
          peps2b = reflection(peps2)
	  eloc = fpeps_util.fdot(peps2b,peps1b)
	  if debug: print '(i,j)=',(i,j),' <U>=',eloc
	  etot += eloc

    #
    # T-term
    #
    tC_aa,tA_aa = fpeps_util.genTaa()
    tC_bb,tA_bb = fpeps_util.genTbb()
    # Jordan-Wigner sign factor for i\=j 
    parity = fpeps_util.genParitySgn([0,1],[2,2])
    tC_aa = einsum('xqp,pr->xqr',tC_aa,parity)
    tC_bb = einsum('xqp,pr->xqr',tC_bb,parity)

    # Vertical bond => Simple is the chosen embedding
    if debug: print "\nVbonds:"
    for j in range(nc):
       for i in range(nr-1):
	  eloc = [0,0]
	  for idx,thops in enumerate([[tC_aa,tA_aa],[tC_bb,tA_bb]]):
	     tC,tA = thops
	     peps2 = peps.copy(peps0)
             peps2[i  ,j] = einsum('xqp,pludr->qlxudr',tC,peps2[i  ,j]).merge_adjpair(2,3)
             peps2[i+1,j] = einsum('xqp,pludr->qluxdr',tA,peps2[i+1,j]).merge_adjpair(3,4)
	     sgn = fpeps_util.genParitySgn(peps2[i+1,j].ptys[1],peps2[i+1,j].dims[1])
	     peps2[i+1,j] = einsum('lL,qLudr->qludr',sgn,peps2[i+1,j])
	     peps2b = reflection(peps2)
	     eloc[idx] = 2.0*fpeps_util.fdot(peps2b,peps1b)
          if debug: print 't<i,j>=',(i,j),'eloc=',sum(eloc),eloc
	  etot += sum(eloc)

    # horizontal bond
    #     |/   |/
    #   --A----B--
    #    /|   /|
    #     *----*
    #     |    |
    if debug: print "\nHbonds:"
    for i in range(nr):
       for j in range(nc-1):
	  for idx,thops in enumerate([[tC_aa,tA_aa],[tC_bb,tA_bb]]):
	     tC,tA = thops
             peps2 = peps.copy(peps0)
             peps2[i,j  ] = einsum('xqp,pludr->qludxr',tC,peps2[i,j  ]).merge_adjpair(4,5)
             peps2[i,j+1] = einsum('xqp,pludr->qxludr',tA,peps2[i,j+1]).merge_adjpair(1,2)
	     sgn = fpeps_util.genParitySgn(peps2[i,j].ptys[2],peps2[i,j].dims[2])
	     peps2[i,j] = einsum('uU,qlUdr->qludr',sgn,peps2[i,j])
	     #??? WHY??? => ABSORBED. => Correct.
	     #for k in range(i):
	     #   sgn = fpeps_util.genParitySgn(peps2[k,j+1].ptys[0],peps2[k,j+1].dims[0])
	     #   peps2[k,j+1] = einsum('qQ,Qludr->qludr',sgn,peps2[k,j+1])
	     peps2b = reflection(peps2)
	     eloc[idx] = 2.0*fpeps_util.fdot(peps2b,peps1b)
          if debug: print 't<i,j>=',(i,j),'eloc=',sum(eloc),eloc
	  etot += sum(eloc)

    return ovlp,etot

#==============================================================================

def test_plot2by2N4():
    elst = np.loadtxt('data/peps2by2N4.dat')
    import matplotlib.pyplot as plt
    efci = -3.0690653
    plt.plot(np.log10(elst-efci),'ro-')
    plt.savefig('data/convergence.pdf')
    plt.show()
    return 0

if __name__ == '__main__':
   #test_save()
   #test_load()
   test_plot2by2N4()
