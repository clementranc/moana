# -*- coding: utf-8 -*-

import ulat.dbc as dbc

def from_parfile(filename: str, path: str = './') -> dbc.Dataset:
    """Load instrument properties from a 'par' file.

    Args:
        filename: name of the parfile.
        path: path to the parfile

    Return:
        A :obj:`ulat.dbc.Dataset` object.

    .. seealso::
        :func:`ulat.dbc.tools.from_parfile`
    """
    dataset = filename.split('par')[1]
    dataset = dbc.Dataset(dataset, path=path)
    return dataset
