import concave_hull
import copy
from .frames import LensReferenceFrame
import numpy as np
import pandas as pd
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


class ResonantCaustic:
    """Sample a resonant caustic.

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
            **kwargs):

        self.sep = sep
        self.q = q
        self.xcoords = np.array([0.0, -sep])
        self.eps = np.array([1.0/(1.0+q), q/(1.0+q)])        
        self.edge = None
        self._ntot = 50

    def _sample(self, ntot, uniform=False):

        nb = int(0.5 * ntot)
        if uniform:
            phi_s = self._curvilinear_length()
            uniform_phi_s = self._make_uniform(phi_s, nb)
            self.phi_s = phi_s
            self.uniform_phi_s = uniform_phi_s
            phi = uniform_phi_s[0]
        else:
            phi_min = 0
            phi_max = 2 * np.pi
            phi = np.linspace(phi_min, phi_max, ntot, endpoint=False)
        
        # Use an angle as parameter to sample the caustic
        sampling = pd.DataFrame()
        sampling['phi'] = phi
        sampling['phi_for_roots'] = phi

        mask = sampling['phi'] > 2*np.pi
        sampling.loc[mask, 'phi_for_roots'] = sampling.loc[mask, 'phi'] - 2*np.pi
        
        # Find the critical points
        z = np.array([critic_2l(self.sep, self.q, a) for a in sampling['phi_for_roots'].values])
        
        # Map the critical curves from the lens to source plane
        zz = np.array([lens_equation_2l(self.xcoords, self.eps, a) for a in z])

        # Store the caustic
        zetasp = ['zeta0', 'zeta1', 'zeta2', 'zeta3']
        zetalp = ['z0', 'z1', 'z2', 'z3']
        for i in range(4):
            sampling[zetasp[i]] = zz.T[i]
            sampling[zetalp[i]] = z.T[i]
        
        # Choose zeta2, zeta3 for Im(zz) > 0
        for i in range(len(sampling)):
            y = copy.copy(sampling)
            x = y.loc[i, zetasp]
            mask = np.argsort(x.to_numpy().imag)
            for j in range(4):
                sampling.loc[i, zetasp[j]] = y.loc[i, zetasp[mask[j]]]

        if not uniform:
            self.edge = self._sort_points_using_concave_hull_algo(sampling)
            sampling.sort_values('phi', ascending=True, inplace=True)
            self.half_caustic = sampling

    def _sort_points_using_concave_hull_algo(self, sampling) -> np.ndarray:
        zetasp = ['zeta2']
        points = sampling[zetasp].to_numpy()
        points = points.reshape((len(sampling), 1)).flatten()
        points = np.array([points.real, points.imag]).T
        self.points = points
        hull = concave_hull.compute(points, 3, iterate=True)
        hull = np.array([hull.T[0] + 1j * hull.T[1]]).flatten()
        return hull


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
    return np.array(z - wk(np.conj(z), xcoords, eps, 1))

def get_dzdphi(sep, q, z):
    s_list = np.array([0.0, -sep])
    eps_list = np.array([1.0/(1.0+q), q/(1.0+q)])
    return -1j * wk(z, s_list, eps_list, 1) / wk(3, s_list, eps_list, z)

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

def wide_limit_2l(q: np.ndarray) -> np.ndarray:
    """Compute the limit between resonant and wide-separation caustics.

    Args:
        q: list of lens mass ratios.

    Returns:
        limit as a function of q.

    """
    cw = (1.0 + q**(1.0 / 3.0))**3 / (1.0 + q)
    dw = np.power(cw, 0.5)
    return dw

def close_limit_2l(q: np.ndarray) -> np.ndarray:
    """Compute the limit between resonant and close-separation caustics.

    Args:
        q: list of lens mass ratios.

    Returns:
        limit as a function of q.

    """
    if np.atleast_1d(q).shape[0] > 1:
        dc = np.array([close_limit(a) for a in q])
    else:
        cc = (1.0 + q)**2 / (27.0 * q)

        coeff = [cc, (1.0 - 3.0 * cc), 3.0 * cc, -cc]
        x4 = np.roots(coeff)
        dc = np.power(x4[np.where(np.abs(x4.imag) < 1e-10)].real[0], 1.0/4.0)
    return dc

def shape(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the limit between resonant and close-separation caustics.

    Args:
        s: list of separation values.
        q: list of lens mass ratios.

    Returns:
        list of str, where 'c' means close, 'r' means 'resonant', and 'w' means
            wide'.

    """
    if np.atleast_1d(s).shape[0] > 1:
        topology = np.array([shape(a, q) for a in s])
    else:
        dc = close_limit_2l(q)
        dw = wide_limit_2l(q)

        if s <= dc:
            topology = 'c'
        elif ((s <= dw) & (s > dc)): 
            topology = 'r'
        else: 
            topology = 'w'

    return topology





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
    for i in range(affix.shape[0]):
            x += mass_fraction[i] / np.power(z-affix[i], k)
    return w * x







