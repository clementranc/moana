from __future__ import annotations

import numpy as np
import pandas as pd
import sys
try:
    from typing import Literal
except:
    from typing_extensions import Literal

class LensReferenceFrame:
    """Reference frame where x-axis is a lens symmetry axis.

    Args:
        center: origin of the frame.
        x_axis: direction of the x-axis. The primary (secondary) is '1' ('2'),
            so '12' means: 'from the primary, to the secondary'.

    """

    hint_frame = Literal['barycenter', 'primary', 'secondary']
    hint_dir = Literal['12', '21']

    def __init__(self, center: hint_frame = 'barycenter', x_axis: hint_dir = '12'):

        self.center = center
        self.x_axis = x_axis

    @property
    def center(self):
        return self._center
    
    @center.setter
    def center(self, value):
        self._center = value
        
        if (not value == 'barycenter') & (not value == 'primary') & (not value == 'secondary'):
            txt = "Argument error: [center = 'barycenter' | 'primary' | 'secondary']."
            sys.exit(txt)

    @property
    def x_axis(self):
        return self._x_axis
    
    @x_axis.setter
    def x_axis(self, value):
        self._x_axis = value
        
        if (not value == '12') & (not value == '21'):
            txt = "Argument error: [x_axis = '12' | '21']."
            sys.exit(txt)

    def to_frame(self, z: np.ndarray, new_frame: LensReferenceFrame, **kwargs):
        """Compute positions in a new reference frame.

        Args:
            z: position in old reference frame.
            new_frame: new reference frame.

        Keyword arguments:
            sep (float): separation in Einstein units.
            gl1 (float): distance from the barycenter to the primary, in Einstein
                units

        """
        z_new = pd.DataFrame()
        z_new['before'] = z
        z_new['after'] = z
        
        sep = kwargs['sep']
        gl1 = kwargs['gl1']
        x_offset = 0

        if not self._center == new_frame.center:
            if self._center == 'primary':
                if new_frame.center == 'secondary':
                    x_offset = sep
                if new_frame.center == 'barycenter':
                    x_offset = np.abs(gl1)

            if self._center == 'secondary':
                if (new_frame.center == 'primary'):
                    x_offset = - sep
                if (new_frame.center == 'barycenter'):
                    x_offset = - sep + np.abs(gl1)

            if self._center == 'barycenter':
                if (new_frame.center == 'primary'):
                    x_offset = - np.abs(gl1)
                if (new_frame.center == 'secondary'):
                    x_offset = sep - np.abs(gl1)
        
        if self._x_axis == new_frame.x_axis:
            if self._x_axis == '12':
                z_new['after'] = z_new['after'] - x_offset
            if self._x_axis == '21':
                z_new['after'] = z_new['after'] + x_offset

        else:
            z_new['after'] = - z_new['after'].values.conjugate()
            if self._x_axis == '12':
                z_new['after'] = z_new['after'] + x_offset

            if self._x_axis == '21':
                z_new['after'] = z_new['after'] - x_offset
        return z_new['after'].values
