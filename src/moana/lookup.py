import glob
import numpy as np
import pandas as pd
import re

class ModelsSummary:
    
    def __init__(self, files=None, fcn=None):
        if not files == None:
            self.rexp = files
            
        if not fcn == None:
            self.fcn = fcn
        else:
            self.fcn = ""
            
        self.describe = pd.DataFrame()
            
        if (not files == None):
            self._load_summary()
            self._create_model()


    def _load_summary(self):

        self._all_models = ""
        files = glob.glob(self.rexp)
        txt = u"FCN=   {:s}".format(self.fcn)
        for i in range(len(files)):
            with open(files[i], 'r') as fin:
                for line in fin:
                    res = re.findall(txt, line)
                    if len(res) > 0: 
                        self._all_models = "{:s}{:s}{:s}".format(
                            self._all_models, files[i], line)
        self._all_models = [a for a in self._all_models.split('\n') if a != ""]

    def _create_model(self):
        
        for i in range(len(self._all_models)):
            modi = re.split("\s+", self._all_models[i], flags=re.UNICODE)
            
            if len(self.describe) == 0:
                self.describe['file'] = [modi[0][:-1]]
                self.describe['chi2'] = [float(modi[2])]
            else:
                tmp = pd.DataFrame()
                tmp['file'] = [modi[0][:-1]]
                tmp['chi2'] = [float(modi[2])]
                self.describe = pd.concat([self.describe, tmp])
        try:
            del tmp
        except NameError:
            pass
        
        self.describe.drop_duplicates(inplace=True)
        self.describe.reset_index(drop=True, inplace=True)
        
    def describe(self):
        return self.describe
    
    def summary(self):
        return self.describe
    
    def best_fit(self):
        mask = self.describe['chi2'] == np.min(self.describe['chi2'])
        return self.describe.loc[mask]
        
