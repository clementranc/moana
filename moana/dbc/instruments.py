# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

class Dataset:
    """Load and store information about instruments, telescopes and filters.
    
    Args:
        dataset: name of the dataset.
        path: path to the dataset.
        
    Examples:

    If dataset='OB151670', then the class loads the file 'parOB151670', and the 
    instruments properties are stored in the attribute :obj:`DBDataset.instruments`. 
    The observations files are, for example, OB151670.ogle.

    .. code-block:: python

        >>> from moana.dataset import Dataset
        >>> parfile = Dataset('OB151670')

    .. seealso::
        :func:`moana.dbc.tools.from_parfile`

    Attributes:
        instruments: :obj:`pandas.DataFrame` with all the parameters of each instrument.
            Keywords are the same as in the par* files.

    """
    def __init__(self, dataset: str, path: str = '.'):
        self.dataset = dataset
        self.path = path
        self.instruments = self._load_params_file()
        self._which_files()
        self._flux_or_mag()

    def _load_params_file(self) -> pd.DataFrame :
        if self.dataset == None:
            sys.exit("An event name must be provided. Exit.")
        fname = f"{self.path}/par{self.dataset}"
        colnames = ['jclr', 'fudge', 'errmin', 'fmin', 'fmax', 'ald',
                    'bld', 'dayoff', 'sfx', 'long', 'lat']
        col = range(len(colnames))
        fmt = {'jclr': np.int64, 
               'fudge': np.float64,
               'errmin': np.float64, 
               'fmin': np.float64, 
               'fmax': np.float64, 
               'ald': np.float64,
               'bld': np.float64, 
               'dayoff': np.float64, 
               'sfx': np.str, 
               'long': np.float64, 
               'lat': np.float64}
        x = pd.read_table(fname, sep='\s+', names=colnames,
                usecols=col, dtype=fmt, skiprows=3)
        for i in range(len(x)):
            x.at[i, 'sfx'] = x.at[i, 'sfx'].replace("'", "")

        return x

    def _which_files(self):
        self.instruments['lcfile'] = [os.path.exists(
            f"{self.path}/lc{self.dataset}.{self.instruments.sfx.values[i]}")
            for i in range(len(self.instruments))]

    def _flux_or_mag(self):
        self.instruments['type'] = 'unknown'

        mask = (9 <= self.instruments.jclr)\
                & (self.instruments.jclr <= 14)
        self.instruments.at[mask, 'data_type'] = 'mag0'
        mask = (15 <= self.instruments.jclr)\
                & (self.instruments.jclr <= 29)
        self.instruments.at[mask, 'data_type'] = 'mag21'
        mask = (30 <= self.instruments.jclr)\
                & (self.instruments.jclr <= 39)
        self.instruments.at[mask, 'data_type'] = 'flux'
        mask = (40 <= self.instruments.jclr)\
                & (self.instruments.jclr <= 49)
        self.instruments.at[mask, 'data_type'] = 'mag21'
        mask = (50 <= self.instruments.jclr)\
                & (self.instruments.jclr <= 59)
        self.instruments.at[mask, 'data_type'] = 'flux'

