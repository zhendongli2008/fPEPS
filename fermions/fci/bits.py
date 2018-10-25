#
# All counts from the first space = 0
#

#
# Basic
#

# testBit() returns a nonzero result, 2**offset, if the bit at 'offset' is one.
def testBit(int_type, offset):
    mask = 1 << offset
    return(int_type & mask)

# setBit() returns an integer with the bit at 'offset' set to 1.
def setBit(int_type, offset):
    mask = 1 << offset
    return(int_type | mask)

# clearBit() returns an integer with the bit at 'offset' cleared.
def clearBit(int_type, offset):
    mask = ~(1 << offset)
    return(int_type & mask)

# toggleBit() returns an integer with the bit at 'offset' inverted, 0 -> 1 and 1 -> 0.
def toggleBit(int_type, offset):
    mask = 1 << offset
    return(int_type ^ mask)

#
# Occupation number representation for the
# Fock space eigenvalue problem: <ON1|H|ON2>c2=E*c1
#
def bitCount(int_type):
   count = 0
   while(int_type):
      int_type &= int_type - 1
      count += 1
   return(count)

def bitCountN(integer,n):
   # Cut the first n strings from integer [count from 0]
   refstr='1'*(n+1)
   refint=int(refstr,2)
   fnlstr=(integer & refint)
   return bitCount(fnlstr)

def action(on,i,itype):
   if itype==1:
      return cre(on,i)
   elif itype==0:
      return ann(on,i)

# ai|ON> 
def ann(on,i):
   # ni=1, gen new vector
   if testBit(on,i):
      if i == 0:
         return clearBit(on,i),1
      else:
	 n = i - 1
	 sgn = (-1)**bitCountN(on,n)
         return clearBit(on,i),sgn
   # ni=0, None 
   else:
      return None

# ai^+|ON>
def cre(on,i):
   # ni=0, gen new vector
   if not testBit(on,i):
      if i == 0:
         return setBit(on,i),1
      else:
	 n = i - 1
	 sgn = (-1)**bitCountN(on,n)
         return setBit(on,i),sgn
   # ni=0, None 
   else:
      return None

def toOrblst(k,n):
   lst = []	
   for i in range(k):
      if testBit(n,i): lst.append(i)
   return lst

if __name__ == '__main__':
    n=10
    print bin(n)
    print bin(n).count('1')
    for i in range(10):
       print 'idx=',i,bool(testBit(n,i)),bitCount(i)
       if ann(n,i) != None: print 'ann',bin(ann(n,i)[0]),ann(n,i)[1]
       if cre(n,i) != None: print 'cre',bin(cre(n,i)[0]),cre(n,i)[1]
    # 0b1010
    print bitCountN(n,0) # 0
    print bitCountN(n,1) # 1
    print bitCountN(n,2) # 1
    print bitCountN(n,3) # 2
    print bitCountN(n,4) # 2
