from ptensor.include import np
from ptensor.parray_helper import einsum 
import peps
import peps_h
import autograd
import mc_helper

def test_energy():
    np.random.seed(5)
    nr=4
    nc=4
    pdim=2
    bond=2
    auxbond=20 # None - PHP,PP,PHP/PP,eav= -6.0 1.0 -6.0 -0.375
    def energy_fn(vec, pdim,bond):
        P = peps.aspeps(vec, (nr,nc), pdim, bond)
        PHP = peps_h.eval_heish(P, P, auxbond)
        PP = peps.dot(P,P,auxbond)
	e = PHP/PP
        print ' PHP,PP,PHP/PP,eav=',PHP,PP,e,e/(nr*nc)
        return PHP/PP
    def bound_energy_fn(vec):
        return energy_fn(vec, pdim, bond)
    deriv = autograd.grad(bound_energy_fn)
    # LOAD 
    vec = np.load('peps_vec.npy')
    print 'nparams=',len(vec)
    print bound_energy_fn(vec)
    #print deriv(vec)
    #nparams= 1600
    # PHP,PP,PHP/PP,eav= -2358.79101506 268.806170781 -8.77506274581 -0.548441421613
    #test_basic(vec,nr,nc,pdim,bond)
    test_mc(vec,nr,nc,pdim,bond)
    return 0


def test_basic(vec,nr,nc,pdim,bond):
    P = peps.aspeps(vec, (nr,nc), pdim, bond)
    # Initialization
    auxbond=16 
    configa = np.zeros([nr,nc], dtype=np.int)
    configb = np.zeros([nr,nc], dtype=np.int)
    for i in range(nr):
        for j in range(nc):
            configa[i,j] = (i + j) % 2
            configb[i,j] = (i + j + 1) % 2
    assert np.sum(configa)%2 == 0
    assert np.sum(configb)%2 == 0
    
    x = mc_helper.fromConf2Bits(configa)
    print configa
    print x,bin(x)
    conf = mc_helper.fromBits2Conf(x,nr,nc)
    print conf
    print
    configb = configa.copy() 
    configb[1,1] = 1
    configb[1,2] = 0
    y = mc_helper.fromConf2Bits(configb)
    print configb
    print y,bin(y)

    Pa = peps.create(pdim,configa)
    Pb = peps.create(pdim,configb)
    print '<Pa|P>=',peps.dot(Pa,P,auxbond)
    print '<Pb|P>=',peps.dot(Pb,P,auxbond)
    print 
    print ' configa=\n',configa
    print ' configb=\n',configb
    
    #peps_h.eval_hbond(Pa, Pb, 1, 1, auxbond)
    Hab = peps_h.eval_heish(Pa, Pb, auxbond)
    print ' Hab=',Hab
    assert abs(Hab - 0.5) < 1.e-10
    Haa = peps_h.eval_heish(Pa, Pa, auxbond)
    print ' Haa=',Haa
    assert abs(Haa + 6.0) < 1.e-10
    Hbb = peps_h.eval_heish(Pb, Pb, auxbond)
    print ' Hbb=',Hbb
    assert abs(Hbb + 3.0) < 1.e-10
    return 0


def test_mc(vec,nr,nc,pdim,bond):
    P = peps.aspeps(vec, (nr,nc), pdim, bond)
    # Initialization
    auxbond=20
    configa = np.zeros([nr,nc], dtype=np.int)
    configb = np.zeros([nr,nc], dtype=np.int)
    for i in range(nr):
        for j in range(nc):
            configa[i,j] = (i + j) % 2
            configb[i,j] = (i + j + 1) % 2
    assert np.sum(configa)%2 == 0
    assert np.sum(configb)%2 == 0
   
    def genWs(by):
        cy = mc_helper.fromBits2Conf(by,nr,nc)
        py = peps.create(pdim,cy)
        wy = peps.dot(py,P,auxbond)
	return wy

    mtx = mc_helper.genHeisenbergHlst(nr,nc)
    x = mc_helper.fromConf2Bits(configa)
    wsx = genWs(x)
    nsample = 10000
    maxiter = 30
    sample = [x]*nsample
    ws = np.array([wsx]*nsample)
    iop = 1
    for niter in range(maxiter):
	print '\nniter=',niter
        for i in range(nsample):
	   bx = sample[i]		
           wx = ws[i]
           by = mc_helper.genHeisenbergMove(bx,nr,nc)
           wy = genWs(by)
	   if niter == 0: # accept all
  	      print 'i=',i,bin(by)
	      sample[i] = by
	      ws[i] = wy
           else:
              prob = min([1,(wy/wx)**2])
	      rand = np.random.rand()
	      if prob > rand:
	         sample[i] = by
	         ws[i] = wy 
	      else:
	 	 sample[i] = bx
	         ws[i] = wx
  	      print 'i=',i,prob,rand,'accept=',prob>rand
        # Compute energy
	z = np.sum(ws**2)
	eloc = np.zeros(nsample)
	# E = 1/Z*sum_S |<P|S>|^2*Eloc
	if iop == 0:
	   # Eloc = <P|H|S>/<P|S> (Variational?)
 	   for i in range(nsample):
	      by = sample[i]
              cy = mc_helper.fromBits2Conf(by,nr,nc)
              py = peps.create(pdim,cy)
              PHP = peps_h.eval_heish(P, py, auxbond)
	      eloc[i] = PHP/ws[i]
	      print 'i=',i,eloc[i]
        else:
	   # Eloc = sum_S' <P|S'><S'|H|S>/<P|S> 
	   # We need a way to generate S' from <S'|H|S>
	   for i in range(nsample):
	      sample2 = mc_helper.genHeisenbergConnected(sample[i],nr,nc)
    	      hs = map(lambda s1:mc_helper.genHeisenbergHmn(s1,sample[i],mtx),sample2)
	      wsp = map(lambda s1:genWs(s1),sample2)
	      eloc[i] = np.dot(wsp,hs)/ws[i]
	      print 'i=',i,eloc[i]
	#
        # THIS IS WRONG !!! SIMPLE AVERAGE IS OK.
	#
	esum = np.dot(eloc,ws**2)/z
	print 'etot=',esum,' e_per_site=',esum/(nr*nc)
    return 0


if __name__ == '__main__':
    test_energy()
