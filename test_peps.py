from ptensor.include import np
from ptensor.parray_helper import einsum 
import peps
import peps_h

def test_basic():
    print '[test_basic]'
    np.random.seed(5)
    nr=4
    nc=4
    pdim=2
    bond=1
    auxbond=8
    shape = (nr,nc)
    # simple peps
    pepsc = peps.zeros(shape, pdim, bond)
    pepsc[1,1].prt()
    pepsc[2,2].prt()
    p = einsum('pijkl,pijkl',pepsc[1,1],pepsc[1,1])
    print p
    p = einsum('pijkl,pklmn->ijmn',pepsc[1,1],pepsc[2,2])
    p.prt()
    # random peps
    pepsn = peps.random(shape, pdim, bond)
    pepsn[1,1].prt()
    # add peps
    peps0 = peps.add(pepsn,pepsn)
    pepsn[0,0].prt()
    peps0[0,0].prt()
    peps1 = peps.add(peps0,pepsn)
    peps1[0,0].prt()
    return 0


def test_conf():
    np.random.seed(5)
    nr=3
    nc=3
    pdim=2
    bond=2
    auxbond=4
    # Initialization:
    # [[0 1 0 1]
    #  [1 0 1 0]
    #  [0 1 0 1]
    #  [1 0 1 0]
    #  [0 1 0 1]
    #  [1 0 1 0]]
    configa = np.zeros([nr,nc], dtype=np.int)
    for i in range(nr):
        for j in range(nc):
            configa[i,j] = (i + j) % 2
    pepsa = peps.create(pdim,configa)
    #for i in range(nr):
    #    for j in range(nc):
    #        print '(i,j)=',(i,j)
    #        pepsa[i,j].prt()
    print "CEVAL =",peps.ceval(pepsa,configa,auxbond)
    PP = peps.dot(pepsa,pepsa,None)
    print "<P|P> =", PP
   
    Sp = peps_h.get_Sp()
    Sm = peps_h.get_Sm()
    Sz = peps_h.get_Sz()
    i = j = 0
    
    # Sm*Sp
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("xpq,qludr->plxudr",Sm, pepsa[i,j]  ).merge_adjpair(2,3) 
    pepsb[i,j].prt()
    pepsb[i,j+1] = einsum("xpq,qludr->pluxdr",Sp, pepsa[i,j+1]).merge_adjpair(3,4) 
    valmp = peps.dot(pepsb,pepsa,auxbond)
    print 'valmp=',valmp
    # Sz*Sz
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("pq,qludr->pludr",Sz, pepsa[i,j])
    pepsb[i,j+1] = einsum("pq,qludr->pludr",Sz, pepsa[i,j+1])
    valzz = peps.dot(pepsb,pepsa,auxbond)
    print 'valzz=',valzz
    # Sp*Sm
    pepsb = peps.copy(pepsa)
    pepsb[i,j]   = einsum("xpq,qludr->plxudr",Sp, pepsa[i,j]  ).merge_adjpair(2,3)
    pepsb[i,j+1] = einsum("xpq,qludr->pluxdr",Sm, pepsa[i,j+1]).merge_adjpair(3,4)
    valpm = peps.dot(pepsb,pepsa,auxbond)
    print 'valpm=',valpm
    return 0


def test_energy_grad():
    import autograd
    np.random.seed(5)
    nr=4
    nc=4
    pdim=2
    bond=1
    auxbond=4
    def energy_fn(vec, pdim,bond):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
        PHP = peps_h.eval_heish(P, P, auxbond)
        PP = peps.dot(P,P,auxbond)
	e = PHP/PP
        print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
        return PHP/PP
    def bound_energy_fn(vec):
        return energy_fn(vec, pdim, bond)

    # Add some zeros to make up full bond
    peps0 = peps.random((nr,nc), pdim, bond, fac=1.0) 
    vec = peps.flatten(peps0)
    
    # energy
    e = bound_energy_fn(vec)
    diff = e+0.493730566743
    print 'e=',e
    print 'diff=',diff
    assert abs(diff) < 1.e-8

    # derivative
    print 'vec[4]=',vec[4]
    deriv = autograd.grad(bound_energy_fn)
    print deriv(vec)[4]
    # -0.223730098428

    import scipy.misc
    def bound_energy_fn_x4(xi):
	vec[4] = xi     
        return energy_fn(vec, pdim, bond)
    print scipy.misc.derivative(bound_energy_fn_x4,vec[4],dx=1.e-6,order=3)
    # vec[4]= -0.0231776224103
    # PHP,PP,PHP/PP,eav= -0.353590229788 0.716160622476 -0.493730343013 -0.0308581464383
    # PHP,PP,PHP/PP,eav= -0.35359037912 0.71616060041 -0.493730566743 -0.0308581604215
    # PHP,PP,PHP/PP,eav= -0.353590528452 0.716160578344 -0.493730790473 -0.0308581744046
    # npt=3 -0.223730098559
    # npt=5 -0.223730098621

#    def bound_energy_fn2(vec):
#        def fun(x):
#           P = peps.aspeps(vec, (nr,nc), pdim, bond)
#           return np.log(peps_h.product(P, P, auxbond, x))
#        deriv = autograd.grad(fun)
#        return deriv(1.e-4)
#
#    print 'e=',bound_energy_fn2(vec)
#
#    deriv = autograd.grad(bound_energy_fn)
#    print 'nparams=',len(vec)
#    print bound_energy_fn(vec)
#    print deriv(vec)
#    exit()
    return 0

    
def test_min():
    np.random.seed(5)
    nr=4
    nc=4
    pdim=2
    bond=2
    auxbond=4 # None - PHP,PP,PHP/PP,eav= -6.0 1.0 -6.0 -0.375
    def energy_fn(vec, pdim,bond):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
        PHP = peps_h.eval_heish(P, P, auxbond)
        PP = peps.dot(P,P,auxbond)
	e = PHP/PP
        print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
        return PHP/PP

    # Initialization
    configa = np.zeros([nr,nc], dtype=np.int)
    configb = np.zeros([nr,nc], dtype=np.int)
    for i in range(nr):
        for j in range(nc):
            configa[i,j] = (i + j) % 2
            configb[i,j] = (i + j + 1) % 2
    assert np.sum(configa)%2 == 0
    assert np.sum(configb)%2 == 0
    pepsa = peps.create(pdim,configa)
    pepsb = peps.create(pdim,configb)
    
    # AUTOGRAD
    pepsc = peps.random(pepsa.shape, pdim, bond-1, fac=1.e-2) 
    peps0 = peps.add(pepsa,pepsc)
    peps0 = peps.add_noise(peps0,fac=1.e-1)
    vec = peps.flatten(peps0)

    import scipy.optimize
    import autograd
    def bound_energy_fn(vec):
        return energy_fn(vec, pdim, bond)
    
    deriv = autograd.grad(bound_energy_fn)
    print 'nparams=',len(vec)
    #print bound_energy_fn(vec)
    #print deriv(vec)
 
    def save_vec(vec):
	fname = 'peps_vec1'
	np.save(fname,vec)
	print ' --- save vec into fname=',fname
	return 0

    #vec = np.load('peps_vec.npy')
    #peps0 = peps.aspeps(vec, (nr,nc), pdim, bond)
    #peps0 = peps.add_noise(peps0,fac=5.e-1)
    #vec = peps.flatten(peps0)

    # Optimize
    result = scipy.optimize.minimize(bound_energy_fn, jac=deriv, x0=vec,\
		    		     tol=1.e-4, callback=save_vec)
    P0 = peps.aspeps(result.x, (nr,nc), pdim, bond)
    print "final =",energy_fn(peps.flatten(P0), pdim, bond)
    return 0


if __name__ == '__main__':
   #test_basic()
   #test_conf()
   #test_energy_grad()
   test_min()
