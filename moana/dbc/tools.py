# -*- coding: utf-8 -*-

import moana.dbc as dbc

def from_parfile(filename: str, path: str = './') -> dbc.Dataset:
    """Load instrument properties from a 'par' file.

    Args:
        filename: name of the parfile.
        path: path to the parfile

    Return:
        A :obj:`moana.dbc.Dataset` object.

    """
    dataset = filename.split('par')[1]
    dataset = dbc.Dataset(dataset, path=path)
    return dataset

def mass_fration_to_mass_ratio(x: float) -> float :
    """Compute the planet mass ratio from planet mass ratio.

    Args:
        x: mass fraction. For a two-body problem with masses M1 and M2, the mass
            fraction is x = M2 / (M1 + M2), with M2 < M1.

    Return:
        The corresponding mass ratio, i.e., M2/M1, with M2 < M1.

    """
    return x / (1 - x)