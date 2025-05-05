# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re

class Output:
    """Class that load a model, the corresponding data and residuals.
    
    Args:
        run: name of the run for which the files should be loaded.
        path: path to the output files.

    Attributes:
        idx_data: line number (int) of the first data in resid file.
        param: :obj:`pandas.Series` with model parameters, including flux.
        resid: :obj:`pandas.DataFrame` of the data.

    """
    def __init__(self, run: str, path: str = './', **kwargs):
        self.path = path
        self.run = run
        self.param = None
        if kwargs == None: self.kwargs = dict()
        else: self.kwargs = kwargs
        
    def load(self, resid=True, model=True):
        """Method that load the output files."""
        if self.param == None:
            self.__load_model()
        if resid:
            self.__load_resid()
        if model:
            self.__load_fitlc()
        
    def __load_model(self):
        fname = f'{self.path}/resid.{self.run}'
        file = open(fname, 'r')
        i = 0
        p = dict()
        for line in file:
            l = re.split(r'\s+', line.strip())
            if (i%2 == 0) & (l[0]=='t'): 
                i+=1
                break
            elif i%2 == 0:
                x = l
            elif not i%2 == 0:
                [p.update({x[i]: float(l[i])}) for i in range(len(l))
                        if np.abs(float(l[i])) > 1e-10]
            i+=1
        file.close()

        if not 'Tstar' in p: p.update({'Tstar': 0.0})
        if not 'eps1' in p: p.update({'eps1': 0.0})
        
        self.sfx = [a[2:] for a in p if (a[0:2]=='A0') & (p[a] > 1e-10)]
        self.idx_data = i
        self.param = pd.Series(p)
        self.n_sfx = len(self.sfx)
        self.__compute_missing_params()

    def __load_resid(self):
        fname = f'{self.path}/resid.{self.run}'
        colnames = ['date', 'mgf_model', 'res_mgf', 'sig_mgf', 'chi2', 'jclr', 'sfx']
        col = [0, 1, 2, 3, 4, 5, 6]
        fmt = { 'date': np.float64,
                'mgf_model': np.float64,
                'res_mgf_model': np.float64,
                'sig_mgf': np.float64,
                'chi2': np.float64,
                'jclr': np.int64,
                'sfx': str}
        self.resid = pd.read_table(fname, sep=r'\s+', names=colnames,
                                   usecols=col, dtype=fmt, skiprows=self.idx_data)
        self.sfx = np.unique(self.resid.sfx)
        self.n_sfx = len(self.sfx)
        
        # Use convention from Bennett's code
        # mgf_data --> magnification of data: (flux - fb / fs)
        self.resid['mgf_data'] = - self.resid['res_mgf'] + self.resid['mgf_model']
        # res_mgf --> mgf_data - magnification_of_model
        self.resid['res_mgf'] = - self.resid['res_mgf']
        
    def __load_fitlc(self):

        if 'dataset' in self.kwargs:
            parfa = self.kwargs['dataset']
            colnames = ['date', 'mgf']
            [colnames.append(f'flux_{a}') for a in list(parfa.instruments[parfa.instruments['lcfile']]['sfx'].values)]
            [colnames.append(a) for a in ['xs', 'ys']]
            col = range(len(colnames))
            fmt = dict()
            [fmt.update({a : np.float64}) for a in colnames]
            fname = f'{self.path}/fit.lc_{self.run}'
            self.fitlc = pd.read_table(fname, sep=r'\s+', names=colnames, usecols=col,
                                       dtype=fmt, skiprows=self.idx_data-1)

            # Compute magnification from flux
            ordered_instruments = list(parfa.instruments[parfa.instruments['lcfile']]['sfx'].values)
            for i in range(len(ordered_instruments)):
                instr = ordered_instruments[i]
                try:
                    self.fitlc[f'mgf_{instr}'] = (self.fitlc[f'flux_{instr}'] - self.param[f'A2{instr}']) / self.param[f'A0{instr}']
                except:
                    self.fitlc[f'mgf_{instr}'] = -1

        else:
            n_sfx = self.n_sfx
            fname = f'{self.path}/fit.lc_{self.run}'
            colnames = ['date', 'mgf', 'xs', 'ys']
            col = [0, 1, 2 + n_sfx, 3 + n_sfx]
            fmt = { 'date': np.float64,
                    'mgf_model': np.float64,
                    'xs': np.float64,
                    'ys': np.float64}
            self.fitlc = pd.read_table(fname, sep=r'\s+', names=colnames, usecols=col,
                                       dtype=fmt, skiprows=self.idx_data-1)            

    def compare(self, model):
        
        diff = pd.DataFrame()
        diff['date'] = self.resid.date
        diff.sort_values('date', inplace=True)
        diff['dchi2'] = 0
        diff['sum_dchi2'] = 0
        diff['sfx'] = self.resid.sfx
        
        x = model.resid
        x = x.sort_values('date', inplace=False)
        diff['date2'] =  x['date']
        
        minmax = dict()
        for i in range(len(self.sfx)):
            mask = self.resid['sfx'] == self.sfx[i]
            diff.loc[mask, 'dchi2'] = x.loc[mask, 'chi2']\
                - self.resid.loc[mask, 'chi2']
            diff.loc[mask, 'sum_dchi2'] = [np.sum(
                diff.loc[mask, 'dchi2'].values[:i+1])
                for i in range(len(diff[mask]))]
            
            minmax.update({self.sfx[i]: [np.min(diff.loc[mask, 'sum_dchi2']),
                                        np.max(diff.loc[mask, 'sum_dchi2'])]})

        return diff, minmax

    def __compute_missing_params(self):
        """Compute parameters derived from others."""

        locparam = self.param
        rho = locparam['Tstar'] / locparam['t_E']
        q = locparam['eps1'] / (1.0 - locparam['eps1'])
        self.param = pd.concat([self.param, pd.Series({'rho': rho, 'q': q})])
