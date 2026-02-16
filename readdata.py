# coding: utf-8
# coding: utf-8
from math import *
from numpy import *
from scipy import *
from matplotlib.pyplot import *
matplotlib.interactive(True)

from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))

matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 24
rcParams['ytick.labelsize']=20
rcParams['ytick.direction']='in'
rcParams['ytick.right']='True'
rcParams['xtick.labelsize']=20
rcParams['xtick.direction']='in'
rcParams['xtick.top']='True'
rcParams['figure.figsize']= [8, 6]

# this is the most important line from the header - it loads the python routines to analyse data
from magic import *

s = Surf(ivar=20, ave=False, tag='rot01') # useful plotting routines ivar = 20 donne le dernier pas de temps
s.avg(field='Br', cm='seismic') # plots of averages
s.surf('Br', r=0.9, cm = 'seismic') # surface plots
gr = MagicGraph(ivar=20,tag='rot01') # access to 3D fields 

gr.vtheta # theta component of velocity - numpy array
gr.vtheta.shape # theta component of velocity - shape
gr.Btheta.shape # theta component of magnetic field - shape
gr.radius # radius  - numpy array
gr.radius.shape  
gr.ntheta # number of pts in latitude
gr.nphi # number of pts in azimuth
# you have to "construct" the grid in phi and theta yourself - check the plotting routines 

sp = MagicSpectrum(tag='rot01', field='e_mag', ispec=20) # 1D spectra
sp = MagicSpectrum2D(tag='rot01', field='e_mag', ispec=20) # 2D spectra - throughs an error for now, check with MagIC website
