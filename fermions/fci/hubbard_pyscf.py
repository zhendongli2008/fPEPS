import numpy
import scipy.linalg
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.ao2mo
import pyscf.lib.logger as logger
import math
from pyscf.fci import cistring
from pyscf import fci

mol = pyscf.gto.Mole()

U = 1.5

n = 4
ne_max = 4
bonds = [[0,1],[0,2],[1,3],[2,3]]

n = 9
ne_max = 10
bonds = [[0,1],[1,2],[3,4],[4,5],[6,7],[7,8],[0,3],[3,6],[1,4],[4,7],[2,5],[5,8]]

# 
#       3------7-----11------15
#      /      /      /      /
#     2--*---6--*---10-----14
#    /   |  /   |  /      /
#   1----*-5----*-9------13
#  /     |/     |/      / 
# 0------4------8------12
# 
ll = 2
n = ll**2
ne_max = 4
bonds = []
for i in range(ll):
   ista = i*ll 
   bonds += [[i*ll+j,i*ll+j+1] for j in range(ll-1)]
   bonds += [[i+ll*j,i+ll*(j+1)] for j in range(ll-1)]

mol.nelectron = ne_max
mf = pyscf.scf.RHF(mol)
h1 = numpy.zeros((n,n))
for i,j in bonds:
   h1[i,j] = h1[j,i] = -1.0
#for i in range(n-1):
#    print i, i+1
#    h1[i,i+1] = h1[i+1,i] = -1.0
#h1[n-1,0] = h1[0,n-1] = -1.0 # PBC

eri = numpy.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = U

mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
mf._eri = pyscf.ao2mo.restore(8, eri, n)

e,c = scipy.linalg.eigh(h1)
mf.init_guess = '1e'

# dm0 is the initial guess
mf.kernel()
#mc = pyscf.mcscf.CASCI(mf, n, ne_max)

#site_orb = numpy.eye(len(mc.mo_coeff))
#na = cistring.num_strings(n, n/2.0)
#nb = cistring.num_strings(n, n/2.0)
#numpy.random.seed(0)
#ci0  = (2*numpy.random.rand(na,nb)-1).ravel()
#ci0 /= numpy.linalg.norm(ci0)

#mc.verbose = 4

#emc = mc.casci(mo_coeff=site_orb,ci0=ci0)[0]

mo_coeff = numpy.identity(n)
for nelec in range(2,ne_max+1,2):
   ham = fci.direct_spin1.pspace(h1,eri,n,nelec)[1]
   import scipy.linalg
   e,v = scipy.linalg.eigh(ham)
   print 'nelec=',nelec,'e=',e[:2]
