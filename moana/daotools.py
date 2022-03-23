
import numpy as np
import pandas as pd

class ListStar:
    """
    """
    def __init__(self,
            file: str,
            path: str = './',
            **kwargs):

        self.path = path
        self.name = file
        self.extension = file.split('.')[-1]
        self._load_list_stars()
        
    def _load_list_stars(self):
        """
        """
        filename = f'{self.path}/{self.name}'
        if self.extension == 'lst':
            colnames = ['id', 'x', 'y', 'mag', 'err1', 'err2']
        elif self.extension == 'coo':
            colnames = ['id', 'x', 'y', 'mag', 'err1', 'err2', 'last']
        else:
            colnames = ['id', 'x', 'y', 'mag', 'err1', 'err2']
        col = range(len(colnames))
        fmt = dict()
        [fmt.update({a: np.float64}) for a in colnames]
        fmt.update({'id': np.int64})
        self.table = pd.read_table(filename, sep='\s+', names=colnames, 
                                       usecols=col, dtype=fmt, skiprows=3)
        
        # Load 3 first lines of input file
        three_first_lines = ''
        file = open(filename, 'r')
        for i in range(3):
            three_first_lines = f'{three_first_lines}{file.readline()}'
        file.close()
        self.header = three_first_lines
        
    def to_ds9(self,
               filename: str,
               path: str = None,
               markersize: int = 10,
               markercolor: int = 'green',
               linewidth: float = 1,
               marker: str = 'circle',
               text_offset = [0, 10],
               **kwargs):
        """
        """
        
        if path == None: path = self.path

        opts = dict({
            'marker': marker,
            'markersize': markersize,
            'markercolor': markercolor,
            'linewidth': linewidth,
            'offset': text_offset,
        })
        label_short = ['ms', 'mc', 'lw']
        label_long = ['markersize', 'markercolor', 'linewidth']
        defaults_rank = [1, 2, 3]
        for i in range(len(label_short)):
            if label_short[i] in kwargs:
                if not kwargs[label_short[i]] == self.to_ds9.__defaults__[defaults_rank[i]]:
                    opts.update({label_long[i] : kwargs[label_short[i]]})

        # Create DS9 regions to check the position of the lens and source
        txt = ('# Region file format: DS9 version 4.1\n'
               'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
               'image\n')
        
        for i in range(len(self.table)):
            star = self.table.iloc[[i], [self.table.columns.get_loc('x'), 
                                         self.table.columns.get_loc('y')]].values[0]
            starid = self.table.iloc[[i], [self.table.columns.get_loc('id')]].values[0]
            offset = 25
            textxy = star + [opts['offset'][0], opts['markersize'] + opts['offset'][1]]
            if opts['marker'] == 'point':
                txt = f"{txt}{opts['marker']}({star[0]:.3f},{star[1]:.3f}) # point=x {opts['markersize']} color={opts['markercolor']}\n"
            else:
                txt = f"{txt}{opts['marker']}({star[0]:.3f},{star[1]:.3f},{opts['markersize']}) # width={opts['linewidth']} color={opts['markercolor']}\n"
            txt = f"{txt}text {textxy[0]:.3f} {textxy[1]:.3f} # text={{{starid[0]:d}}} width={opts['linewidth']} color={opts['markercolor']}\n"

        fname = f'{path}/{filename}'
        if len(fname) > 4:
            if not fname[-4:] == '.reg': fname = f'{fname}.reg'
        else: fname = f'{fname}.reg'
        file = open(fname, 'w')
        file.write(txt)
        file.close()


    def to_coo(self,
               filename: str,
               path: str = '.',
               **kwargs):
        """Write a coo file from a list of stars.
        """
        input1 = self.table
        txt = f'{self.header}'
        for i in range(len(input1)):
            txt = (f"{txt}{int(input1['id'].values[i]):7d} {input1['x'].values[i]:8.2f} {input1['y'].values[i]:8.2f} "
                   f"{input1['mag'].values[i]:8.3f} {input1['err1'].values[i]:8.3f} {input1['err2'].values[i]:8.3f} "
                   f"{input1['last'].values[i]:8.3f}\n")

        fn = f'{path}/{filename}'
        print("Create file:")
        file = open(fn, 'w')
        file.write(txt)
        file.close()
        print(f"{fn}")


    def find_my_stars(self,
               my_stars: list,
               inplace: bool = False,
               **kwargs):
        """Cross-match with a table of star coordinates.


            my_stars = [[1093.82, 1171.43, 'Unrelated South West'],
                        [1108.85, 1166.67, 'Unrelated South East'],
                        [1112.51, 1183.30, 'Target']]


        """
        if not inplace:
            input1 = self.table.copy()
        else:
            input1 = self.table

        input1['distance'] = input1['x']
        for i in range(len(my_stars)):
            input1['distance'] = np.sqrt(np.power(input1['x'] - my_stars[i][0], 2) + np.power(input1['y'] - my_stars[i][1], 2))
            mask = input1['distance'] == np.min(input1['distance'])

            if my_stars[i][2] == "Target":
                tmp = input1.sort_values('distance')
                print('TARGET (Lens + Source): distances are in pixels')
                print(tmp[['id', 'x', 'y', 'distance']].head(4).to_string(index=False))
            else:
                print(f"{my_stars[i][2]:20} --> {input1.loc[mask, 'id'].values[0]:7d}  [distance: {np.min(input1['distance']):.1e} pixels]")

        if not inplace:
            return input1

def load_star_list_file(file_name = 'image.lst', path = ".", show=1, header=3):
    """Load a list of file, convention of DAOPHOT-II *.lst files."""

    # --- PARAMETERS ---
    file_input1 = f'{path}/{file_name}'
    colnames = ['id', 'x', 'y', 'mag', 'err1', 'err2']
    col = range(len(colnames))
    fmt = dict()
    [fmt.update({a: np.float64}) for a in colnames]
    fmt.update({'id': np.int64})
    input1 = pd.read_table(file_input1, sep='\s+', names=colnames, 
                               usecols=col, dtype=fmt, skiprows=header,
                               comment='#')

    if show > 0:
        print(f'Input list of stars in file {file_input1}:')
        print(input1.head(show))

    # Load 3 first lines of input file
    three_first_lines = ''
    file = open(file_name, 'r')
    for i in range(header):
        three_first_lines = f'{three_first_lines}{file.readline()}'
    file.close()

    return input1, three_first_lines

def cross_match_files(input1, input2):
    """Cross-match two tables of stars based on pixel coordinates."""

    # Cross-matching stars
    print('Cross-matching the two tables of stars...')
    print(' Old ID      New ID')
    input1['match'] = 0
    input2['distance'] = input2['x']
    for i in range(len(input1)):
        input2['distance'] = np.power(input1.loc[i, 'x'] - input2['x'], 2) + np.power(input1.loc[i, 'y'] - input2['y'], 2)
        mask = input2['distance'] == np.min(input2['distance'])
        input1.loc[i, 'match'] = input2.loc[mask, 'id'].values[0]
        print(f"{input1.loc[i, 'id']:7d} --> {input2.loc[mask, 'id'].values[0]:7d}")