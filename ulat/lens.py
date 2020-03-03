from .frames import LensReferenceFrame
import numpy as np
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


# FUNCTIONS FOR 2-BODY LENSES
# ===========================

def critic_2l(
        s: float,
        q: float,
        phi: float):
    """Cumpute solution of the Witt equation.

    To sample critic curves, use the convention:
    - the heaviest body (mass m1) is the origin;
    - the lightest body (mass m2) is at (-s, 0).

    Args:
        s: separation
        q: the lens mass ratio q = m2/m1.
        phi: the angle parameter in [0,2*pi].

    Returns:
        numpy array of the complex roots.
    """
    coefs = [1, 2 * s, s ** 2 - np.exp(1j * phi),
             -2 * s * np.exp(1j * phi) / (1 + q),
             -(s ** 2 * np.exp(1j * phi) / (1 + q))]
    result = np.roots(coefs)
    return result

def lens_equation_2l(
    xcoords: float, 
    eps: float, 
    z: complex):
    """Apply the binary-lens equation to an affix.

    Conventions:
    - the heaviest body (mass m1) is the origin;
    - the lightest body (mass m2) is at (-s, 0).

    Args:
        s: separation
        q: the lens mass ratio q = m2/m1.
        z: complex number in the lens plane.

    Returns:
        numpy array of the corresponding position in the source plane
    """
    return np.array(z - wk(1, xcoords, eps, np.conjugate(z)))

def get_dzdphi(sep, q, z):
    s_list = np.array([0.0, -sep])
    eps_list = np.array([1.0/(1.0+q), q/(1.0+q)])
    return -1j * wk(2, s_list, eps_list, z) / wk(3, s_list, eps_list, z)

def get_dzeta_phi(s, q, z):
    #offset = z + s
    dz_dphi = get_dzdphi(s, q, z)
    dzeta_dphi = dz_dphi - np.conj(wkl2(2, s, q, z) * dz_dphi)
    
    d2zeta_dphi2 = - np.conj(wkl2(3, s, q, z) * dz_dphi * dz_dphi)
    d2zeta_dphi2 = d2zeta_dphi2 - wkl2(2, s, q, z) / wkl2(4, s, q, z)
    
    d3zeta_dphi2 = - 1j * wkl2(2, s, q, z) / wkl2(5, s, q, z)
    d3zeta_dphi2 = d3zeta_dphi2 - np.conj(wkl2(4, s, q, z) * np.power(dz_dphi, 3))
    d3zeta_dphi2 = d3zeta_dphi2 - 2 * np.conj(wkl2(3, s, q, z) * dz_dphi * d2zeta_dphi2)
    
    return dzeta_dphi, d2zeta_dphi2, d3zeta_dphi2

def solve_lens_equation_2l(sep, q, x):
    y = np.atleast_1d(x)
    z = [critic_2l(sep, q, a) for a in y]
    return np.array([lel2(sep, q, zc) for zc in z])







# N-lens equations

def wk(k, s, epsq, z):
    n = k - 1
    w = np.power(-1, n) * np.math.factorial(n)
    sep = np.atleast_1d(s)
    eps = np.atleast_1d(epsq)
    x = 0
    for i in range(len(sep)):
        x += eps[i] / np.power(z-sep[i], k)
    w = w * x
    return w

def wk(z: complex,
    affix: np.ndarray,
    mass_fraction: np.ndarray,
    k: int):
    """Evaluate the function W_k(z) (see Cassan, 2017).

    The complex function W_k(z) is used to compute the lens equation, the
    caustics and many other microlensing quantities, such as derivatives. The definition
    used does not depend on the reference frame, nor the number of point lenses. Let's
    assume that we have N point lenses in what follows.

    Args:
        z: affix of the point where W_k(z) is evaluated.
        affix: array (shape: N, type: complex) of the affix of each point
            lenses.
        mass_fraction: array (shape N, type: float) with the corresponding mass
            fractions.
        k: order of the function W_k(z).

    """
    n = k - 1
    w = np.power(-1, n) * np.math.factorial(n)
    if not affix.shape[0] == mass_fraction.shape[0]:
        sys.exit("Error: not the same number of mass fractions and lenses.")
    x = 0
    for i in range(affixes.shape[0]):
            x += mass_fraction[i] / np.power(z-affixes[i], k)
    return w * x







