import numpy
import scipy.linalg
import time

class mask:
   def __init__(self,info,mvp):
      self.info = info
      self.mvp = mvp 
   def matvec(self,vec):
      return self.mvp(vec,self.info)

#
# LOBPCG solver with Davidson precondition
#
class eigenSolver:
    def __init__(self):
        self.maxcycle =200
        self.crit_e   =1.e-10
        self.crit_vec =1.e-8
        self.crit_indp=1.e-12
        self.crit_demo=1.e-10
        # Basic setting
        self.iprt  = 1
        self.ndim  = 0
        self.neig  = 0
        self.diag  = None
        self.matvec= None
        self.noise = True

    #@profile
    def matvecs(self,vbas):
        n = vbas.shape[0]
	self.nmvp += n
        wbas = numpy.zeros((n,self.ndim))
        for i in range(n):
           wbas[i] = self.matvec(vbas[i])
        return wbas
 
    #@profile
    def genV0(self,neig):
        # Break the degeneracy artifically in generating v0
	diag = self.diag + 1.e-12*numpy.arange(1.0*self.ndim)/float(self.ndim)
	index = numpy.argsort(diag)[:neig]
        v0 = numpy.zeros((neig,self.ndim),dtype=self.diag.dtype)
	v0[range(neig),index]=1.0
        return v0

    #@profile
    def solve_iter(self,v0=None,iop=4,ifplot=False):
	if self.iprt>0:    
	   print     
           print '-------------------------------'
	   print '  Davidson solver for AX=wX    '
           print '-------------------------------'
	   print '  iop  = ',iop
           print '  ndim = ',self.ndim
           print '  neig = ',self.neig
	   print '  maxcycle = ',self.maxcycle
           print '-------------------------------'
	# Clear counter
	self.nmvp = 0
        #
        # Generate initial guess - input v0 is an np.array (neig,ndim)
	#
	t0 = time.time()
	if v0 == None:
           vbas = self.genV0(self.neig)
        else:
	   vbas = v0.copy()
	# Add random noise to interact with the whole space
	if self.noise: vbas = vbas \
			    + 1.e-3*numpy.random.uniform(0.5,1,size=(self.neig,self.ndim))
	genOrthoBas(vbas,self.crit_indp)
        wbas = self.matvecs(vbas)
        #
        # Begin to solve
        #
        ifconv= False
        neig  = self.neig
        eigs  = numpy.zeros(neig)
	ndim  = neig
	rnorm = numpy.zeros((self.neig,self.maxcycle))
	eigval= numpy.zeros((self.neig,self.maxcycle))
        for niter in range(1,self.maxcycle):
           if self.iprt > 0: 
              print '\n ----------  niter,ndim0,ndim,nmvp =',\
	      	    (niter,self.ndim,ndim,self.nmvp),' ----------'
	   #
	   # Check othonormality of basis 
	   #
	   iden = numpy.dot(vbas,vbas.T)
	   diff = numpy.linalg.norm(iden-numpy.identity(ndim))
	   if diff > 1.e-10:
	      print ' diff_VBAS=',diff
	      print iden
	      exit(1)
           tmpH = numpy.dot(vbas,wbas.T)
    	   eig,vr = scipy.linalg.eigh(tmpH)
	   # CHECK ORTHOGONALITY:
	   vr = vr[:,:neig].transpose(1,0).copy()
	   over = numpy.dot(vr,vr.T)
	   diff = numpy.linalg.norm(over-numpy.identity(neig))
	   if diff > 1.e-10: 
	      print ' diff_VEC=',diff
	      print over
	      exit(1)
	   teig = eig[:neig]
 	 	 
  	   # Eigenvalue convergence
  	   nconv1 = 0
  	   for i in range(neig):
  	      tmp = teig[i]-eigs[i]
	      if self.iprt > 0: 
		 print '  i =',i,' eold=',eigs[i],' enew=',teig[i],' ediff=',tmp
  	      if abs(tmp) <= self.crit_e: nconv1+=1
	   if self.iprt > 0: print ' No. of converged eigval:',nconv1,'/',neig
  	   if nconv1 == neig: 
	      if self.iprt > 0:
		 print ' Cong: all eignvalues converged ! '
  	   eigs = teig.copy()
 	   eigval[:,niter-1] = teig.copy()
  
           # Full Residuals: Res[i]=Res'[i]-w[i]*X[i]
  	   rbas = numpy.dot(vr,vbas)
  	   rbas = numpy.dot(vr,wbas) - numpy.dot(numpy.diag(eigs),rbas)
  	   nconv2 = 0
  	   rindx  = []
           rconv  = [0]*neig
	   for i in range(neig):
	      tmp = numpy.linalg.norm(rbas[i,:])
	      rnorm[i,niter-1] = tmp
  	      if tmp <= self.crit_vec:
  	         nconv2 +=1
  	         rconv[i]=(True,tmp)
              else:
  	         rconv[i]=(False,tmp)  
		 rindx.append(i)     
	      if self.iprt > 0: print '  i,rnorm=',i,rconv[i]
	   if self.iprt > 0: print ' No. of converged eigvec:',nconv2,'/',neig
  	   if nconv2 == neig: 
	      if self.iprt > 0:
		 print ' Cong: all eignvectors converged ! '
 
	   t1 = time.time()
           if self.iprt > -1:
	      if niter == 1: 
		  print ' iter  dim  nmvp  ieig        eigenvalue       rnorm     time/s  '
		  print ' ----------------------------------------------------------------'
	      for i in range(neig):
		  print '%4d %4d %5d %3d %2s %20.12f %10.3e %9.2e'%(niter,ndim,self.nmvp,\
			i,str(rconv[i][0])[0],eigval[i,niter-1],rconv[i][1],t1-t0)
	   t0 = time.time()

   	   # Convergence by either criteria
           ifconv = (nconv1 == neig) or (nconv2 == neig)
	   ebas = numpy.dot(vr,vbas)
  	   if ifconv:
	      if self.iprt > 0: 
 		 print ' Cong: ALL are converged !\n'		
   	      break		
  
           ## Rotated basis to minimal subspace that
  	   ## can give the exact [neig] eigenvalues
	   ## Also, the difference vector = xold - xnew as correction 
	   pr = numpy.identity(ndim)[:neig,:] - vr
	   nindp1,vr2 = dvdson_ortho(vr,pr[rindx,:],self.crit_indp)
	   if nindp1 !=0: vr = numpy.vstack((vr,vr2))
	   nindp1 = 0
	   vbas = numpy.dot(vr,vbas)
	   wbas = numpy.dot(vr,wbas)
	   
           # New directions from residuals
	   for i in range(neig):
  	      if rconv[i][0] == True: continue
     	      # Various PRECONDITIONER:
	      if iop == 0:
		 # gradient
		 pass
	      elif iop == 1:
	         # Davidson
		 tmp = self.diag - eigs[i]
 	         tmp[abs(tmp)<self.crit_demo] = self.crit_demo
	         rbas[i,:] = rbas[i,:]/tmp
 	      elif iop == 2:
	         # Olsen's algorithm works for close diag ~ H : 0.00067468 [3]
	         tmp = self.diag - eigs[i]
 	         tmp[abs(tmp)<self.crit_demo] = self.crit_demo	
	         e1 = numpy.dot(vbas[i,:],rbas[i,:]/tmp)/numpy.dot(vbas[i,:],vbas[i,:]/tmp)
	         rbas[i,:] = -(rbas[i,:]-e1*vbas[i,:])/tmp
	      elif iop == 3:
		 # ABS     
		 tmp = abs(self.diag - eigs[i])
 	         tmp[abs(tmp)<self.crit_demo] = self.crit_demo	
	         rbas[i,:] = rbas[i,:]/tmp
	      elif iop == 4:
		 # ABS+LEVEL-SHIFT ~ Davidson+Gradient
		 tmp = abs(self.diag - eigs[i]) + 0.5
	         rbas[i,:] = rbas[i,:]/tmp

	   # Re-orthogonalization and get Nindp
	   nindp2,vbas2 = dvdson_ortho(vbas,rbas[rindx,:],self.crit_indp)

	   nindp = nindp1 + nindp2
	   if self.iprt > 0: print ' final nindp = ',nindp
           if nindp != 0:
              wbas2 = self.matvecs(vbas2)
              vbas  = numpy.vstack((vbas,vbas2))
              wbas  = numpy.vstack((wbas,wbas2))
	      ndim  = vbas.shape[0]
           else:
	      print 'Convergence failure: unable to generate new direction: Nindp=0 !'
              #exit(1)
	  
        if not ifconv:
           print 'Convergence failure: out of maxcycle !'
           #exit(1)

	if ifplot:
           import matplotlib.pyplot as plt
	   plt.plot(range(self.ndim),self.diag)
           plt.show()
	   plt.savefig("diag.png")

	   for i in range(self.neig):
 	      plt.plot(range(1,niter+1),numpy.log10(rnorm[i,:niter]),label=str(i+1))
	   plt.legend()  
	   plt.savefig("res_conv.png")
           plt.show()
        
	   for i in range(self.neig):
 	      plt.plot(range(1,niter+1),eigval[i,:niter],label=str(i+1))
	   plt.legend()  
	   plt.savefig("eig_conv.png")
           plt.show()

	return eigs,ebas,self.nmvp


def dvdson_ortho(vbas,rbas,crit_indp):
    debug = False
    if debug: print ' Orthogonalization:'
    ndim  = vbas.shape[0] 
    nres  = rbas.shape[0]
    nindp = 0
    vbas2 = numpy.zeros(rbas.shape,dtype=rbas.dtype)
    maxtimes = 5
    for k in range(maxtimes):
       rbas = rbas - reduce(numpy.dot,(rbas,vbas.T,vbas))
    for i in range(nres): 
       rvec = rbas[i,:].copy()	    
       rii  = numpy.linalg.norm(rvec)
       if rii <= crit_indp: continue
       if debug: print '  i,rii=',i,rii
       # NORMALIZE
       rvec = rvec / rii
       rii  = numpy.linalg.norm(rvec)
       rvec = rvec / rii
       vbas2[nindp] = rvec
       nindp = nindp +1
       # Substract all things
       for k in range(maxtimes):
          rbas[i:,:]=rbas[i:,:]-reduce(numpy.dot,(rbas[i:,:],vbas.T,vbas))
          rbas[i:,:]=rbas[i:,:]-reduce(numpy.dot,(rbas[i:,:],
                   		       vbas2[:nindp,:].T,vbas2[:nindp,:]))
    vbas2 = vbas2[:nindp]
    # iden
    if debug and nindp != 0:	   
       tmp = numpy.vstack((vbas,vbas2)) 	 	    
       iden = numpy.dot(tmp,tmp.T)
       diff = numpy.linalg.norm(iden-numpy.identity(iden.shape[0]))
       if diff > 1.e-10:
          print ' error in mgs_ortho: diff=',diff
          print iden
          exit(1)
       else:
          print ' final nindp from mgs_ortho =',nindp,' diffIden=',diff	    
    return nindp,vbas2

def genOrthoBas(vbas,crit_indp):
   vbas[0] = vbas[0]/numpy.linalg.norm(vbas[0])
   nbas = vbas.shape[0]
   if nbas != 1: 
      nindp,vbas2 = dvdson_ortho(vbas[0:1],vbas[1:],crit_indp)
      if nindp + 1 == nbas:
	 vbas[1:] = vbas2
      else:
	 print 'error: insufficient orthonormal basis: nbas/nindp',nbas,nindp+1
         exit()
