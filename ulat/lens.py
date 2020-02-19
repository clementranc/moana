from .frames import LensReferenceFrame
from typing import Optional


class Microlens:
    """Define a binary lens.

    Args:
	sep: separation between primary and secondary, in units of the Einstein radius.
	q: planet-to-host star mass ratio.

    Keyword arguments:
	s (float): alias for sep.
	eps1 (float): planet-to-host star mass fraction. 

    """

    def __init__(self,
            sep: float = 1.0, 
            q: float = 1e-3, 
            frame: Optional[LensReferenceFrame] = None, 
            **kwargs):

        self.q = q
        if 'eps1' in kwargs:
            self.eps1 = kwargs['eps1']
            self.q = self.__eps1_to_q(kwargs['eps1'])
        self.sep = sep
        if 's' in kwargs:
            self.sep = kwargs['s']
        self.__find_cm()

        if frame == None:
            frame = LensReferenceFrame()

        #self.caustic = Caustic(self)

    @property
    def la(self):
        return self._la

    @la.setter
    def la(self, value):
        self._la = value
        
    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, value):
        self._lb = value
        
    def __eps1_to_q(self, eps1):
        return eps1 / (1.0 - eps1)

    def __find_cm(self):
        self._gl1 = - self.sep * self.q / (1.0 + self.q)
        self._gl2 = self.sep / (1.0 + self.q)
        
#    def sample_caustic(self, output=False, *args, **kwargs):
#        if not 'frame' in kwargs:
#            frame = kwargs.update({'frame': LensReferenceFrame(center='barycenter', x_axis='12')})
#        self.caustic.sample(*args, **kwargs)
#        if output:
#            return self.caustic.edge
