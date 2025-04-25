#import concave_hull
import copy
from .frames import LensReferenceFrame
import math
import numpy as np
import pandas as pd
import sys
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
                sampling.loc[i, zetalp[j]] = y.loc[i, zetalp[mask[j]]]

        # Find point C
        z_C = sampling.at[0, 'zeta3']
        zeo1_C = get_dzeta_phi(self.sep, self.q, z_C)

        # Find point A and B
        z_A = sampling.at[0, 'zeta1']
        z_B = sampling.at[0, 'zeta2']
        zeo1_A = get_dzeta_phi(self.sep, self.q, z_A)
        if z_A.real > z_B.real:
            z_A = sampling.at[0, 'zeta2']
            z_B = sampling.at[0, 'zeta1']
            zeo1_A = get_dzeta_phi(self.sep, self.q, z_A)
            bcurr = 'zeta2'
        else:
            bcurr = 'zeta1'

        # Compute Taylor approximation
        x = np.array([get_dzeta_phi(self.sep, self.q, zp) for zp in sampling.z2.values])
        sampling['ze2o1'] = x.T[0]
        sampling['ze2o2'] = x.T[1]
        x = np.array([get_dzeta_phi(self.sep, self.q, zp) for zp in sampling.z3.values])
        sampling['ze3o1'] = x.T[0]
        sampling['ze3o2'] = x.T[1]

        # Prediction
        dphi = np.roll(sampling['phi'].to_numpy(), -1) - sampling['phi'].to_numpy()
        sampling['ze2t'] = sampling['zeta2'] + sampling['ze2o1'] * dphi\
            + 0.5 * sampling['ze2o2'] * dphi**2
        sampling['ze3t'] = sampling['zeta3'] + sampling['ze3o1'] * dphi\
            + 0.5 * sampling['ze3o2'] * dphi**2

        sampling['dsze2'] = np.abs(sampling['ze2o1']) * dphi
        sampling['dsze3'] = np.abs(sampling['ze2o1']) * dphi

        x = (sampling['ze2t'] - sampling['zeta2']).to_numpy()
        sampling['dze2a'] = np.angle(x, deg=True)
        x = (sampling['ze3t'] - sampling['zeta3']).to_numpy()
        sampling['dze3a'] = np.angle(x, deg=True)

        sampling['22'] = np.abs(back_in_pipi(
            np.roll(sampling['dze2a'], -1) - sampling['dze2a']))
        sampling['23'] = np.abs(back_in_pipi(
            np.roll(sampling['dze3a'], -1) - sampling['dze2a']))
        sampling['32'] = np.abs(back_in_pipi(
            np.roll(sampling['dze2a'], -1) - sampling['dze3a']))
        sampling['33'] = np.abs(back_in_pipi(np.abs(
            np.roll(sampling['dze3a'], -1) - sampling['dze3a'])))

        sampling['flag2223'] = (np.abs(sampling['22']-sampling['23']) > 20)\
                & (np.abs(sampling['ze2o1']) > 1e-2)
        sampling['flag3332'] = (np.abs(sampling['33']-sampling['32']) > 20)\
                & (np.abs(sampling['ze3o1']) > 1e-2)
        sampling['flag2232'] = (np.abs(sampling['22']-sampling['32']) > 20)\
                & (np.abs(sampling['ze2o1']) > 1e-2)
        sampling['flag3323'] = (np.abs(sampling['33']-sampling['23']) > 20)\
                & (np.abs(sampling['ze3o1']) > 1e-2)

        sampling['flag2to2'] = (sampling['22'] < 20) & sampling['flag2223']
        sampling['flag2to3'] = (sampling['23'] < 20) & sampling['flag2223']

        sampling['flag3to3'] = (sampling['33'] < 20) & sampling['flag3332']
        sampling['flag3to2'] = (sampling['32'] < 20) & sampling['flag3332']

        b1 = pd.DataFrame()
        b1['zeta'] = [z_A]
        b1['zeo1'] = [zeo1_A]
        b1['phi'] = [0.0]

        b2 = pd.DataFrame()
        b2['zeta'] = [z_C]
        b2['zeo1'] = [zeo1_C]
        b2['phi'] = [2*np.pi]

        bcurr = 1

        cols = [['flag2to2', 'zeta2', 'flag2to3', 'zeta3', 'ze2t'],
                ['flag3to3', 'zeta3', 'flag3to2', 'zeta2', 'ze3t']]
        for i in range(len(sampling)-1):
            if i==0: continue

            b1tmp = pd.DataFrame()
            b2tmp = pd.DataFrame()

            # Branch CB
            if sampling.at[i, cols[bcurr][0]]:
                b1tmp['zeta'] = [sampling.at[i+1, cols[bcurr][3]]]
                b2tmp['zeta'] = [sampling.at[i+1, cols[bcurr][1]]]
                b1tmp['phi'] = [sampling.at[i+1, 'phi']]
                b2tmp['phi'] = [sampling.at[i+1, 'phi'] + 2*np.pi]
            elif sampling.at[i, cols[bcurr][2]]: 
                b1tmp['zeta'] = [sampling.at[i+1, cols[bcurr][1]]]
                b2tmp['zeta'] = [sampling.at[i+1, cols[bcurr][3]]]
                b1tmp['phi'] = [sampling.at[i+1, 'phi']]
                b2tmp['phi'] = [sampling.at[i+1, 'phi'] + 2*np.pi]
                if bcurr==1: bcurr = 0
                else: bcurr = 1
            else:
                d33 = np.abs(sampling.at[i, cols[bcurr][4]] - sampling.at[i+1, cols[bcurr][1]])
                d32 = np.abs(sampling.at[i, cols[bcurr][4]] - sampling.at[i+1, cols[bcurr][3]])

                if d33 < d32:
                    b1tmp['zeta'] = [sampling.at[i+1, cols[bcurr][3]]]
                    b2tmp['zeta'] = [sampling.at[i+1, cols[bcurr][1]]]
                    b1tmp['phi'] = [sampling.at[i+1, 'phi']]
                    b2tmp['phi'] = [sampling.at[i+1, 'phi'] + 2*np.pi]
                else:
                    b1tmp['zeta'] = [sampling.at[i+1, cols[bcurr][1]]]
                    b2tmp['zeta'] = [sampling.at[i+1, cols[bcurr][3]]]
                    b1tmp['phi'] = [sampling.at[i+1, 'phi']]
                    b2tmp['phi'] = [sampling.at[i+1, 'phi'] + 2*np.pi]
                    if bcurr==1: bcurr = 0
                    else: bcurr = 1

            b1 = pd.concat([b1,b1tmp], sort=False)
            b2 = pd.concat([b2,b2tmp], sort=False)

        b2tmp = pd.DataFrame()
        if z_B.imag >= 0:
            b2tmp['zeta'] = [z_B]
            b2tmp['phi'] = [2*np.pi]
        b2 = pd.concat([b2,b2tmp], sort=False)

        b1.reset_index(inplace=True, drop=True)
        b2.reset_index(inplace=True, drop=True)
        self.b1 = b1
        self.b2 = b2

        full = pd.DataFrame()
        full = pd.concat([b1, b2])
        full['ds'] = np.abs((np.roll(full['zeta'].to_numpy(), -1) - full['zeta'])\
                / (np.roll(full['phi'].to_numpy(), -1) - full['phi']))
        full['s'] = np.cumsum(full['ds'].to_numpy())
        full['s'] = full['s'] / full['s'].values[-1]

        self.full = full

        if not uniform:
            #self.edge = self._sort_points_using_concave_hull_algo(sampling)
#            sampling.sort_values('phi', ascending=True, inplace=True)
            self.sampling = sampling


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

def get_dzeta_phi(s, q, z):

    s_list = np.array([0.0, -np.abs(s)])
    eps_list = np.array([1.0/(1.0+q), q/(1.0+q)])
    w_2 = wk(z, s_list, eps_list, 2)
    w_3 = wk(z, s_list, eps_list, 3)
    w_4 = wk(z, s_list, eps_list, 4)
    # w_5 = wk(z, s_list, eps_list, 5)
    # w_6 = wk(z, s_list, eps_list, 6)

    dz_dphi = -1j * w_2 / w_3
    dzeta_dphi = dz_dphi - np.conj(w_2 * dz_dphi)

    d2z_dphi2 = w_2 / w_4
    d2zeta_dphi2 = d2z_dphi2 - np.conj(w_3 * np.power(dz_dphi,2))

    # d3z_dphi3 = - 1j * w_2 / w_5
    # d3zeta_dphi2 = d3z_dphi3 - np.conj(w_4 * np.power(dz_dphi, 2))
    # d3zeta_dphi2 = d3zeta_dphi2 - 2 * np.conj(w_3 * dz_dphi * d2z_dphi2)

    # d4z_dphi4 = - w_2 / w_6
    # d4zeta_dphi4 = d4z_dphi4 - np.conj(w_5 * np.power(dz_dphi,3))
    # d4zeta_dphi4 = d4zeta_dphi4 - 4 * np.conj(w_4 * d2z_dphi2 * dz_dphi)
    # d4zeta_dphi4 = d4zeta_dphi4 - 2 * w_3 * (d3z_dphi3 * dz_dphi + np.power(d2z_dphi2,2))
    
    return dzeta_dphi, d2zeta_dphi2 #, d3zeta_dphi2, d4zeta_dphi4

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
        dc = np.array([close_limit_2l(a) for a in q])
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

def wkl2(k, s, q, z):
    n = k - 1
    w = ( np.power(-1, n) * np.math.factorial(n) / (1 + q) )\
        * ( 1.0 / np.power(z, k) + q / np.power(z + s, k) )
    return w



# N-lens equations

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


# GENERAL FUNCTIONS

def back_in_pipi(x):
    if np.atleast_1d(x).shape[0] > 1:
        return np.array([back_in_pipi(a) for a in x])
    if x > 180:
        return back_in_pipi(x - 360)
    elif x < -180:
        return back_in_pipi(x + 360)
    else:
        return x




