# Anything operate with null like operatoring zero! 

class Null(object):
    """
    Zero placeholder class 
    for use in sparse object arrays.
    Behaves like a generalized zero.
    """
    def __init__(self):
        self.shape = 0

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __sub__(self, other):
        return -other

    def __rsub__(self, other):
        return other
        
    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __div__(self, other):
        return self

    def __ldiv__(self, other):
        return RuntimeError

    def __neg__(self):
        return self

    def __repr__(self):
        return "'0'"

    def __sqrt__(self):
        return self

NULL = Null()
