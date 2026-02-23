""" 
in Pre_main_sequence_star_treatment_simu
git add readdata.py
git commit -m "modifications"
git push

in meso psl 
git pull
"""

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
a = input("dossier voulu dans AnelasticCouette : ")
my_path = '/travail/dynconv/multiscale_dyno/anelasticCouette/'+a

gr = MagicGraph(datadir=my_path,ivar=9,tag='rot01') # access to 3D fields 
print(gr.__dict__.keys())

s = Surf(datadir=my_path,ivar=9, ave=False, tag='rot01') # useful plotting routines ivar = 20 donne le dernier output

s.avg(field='cr', cm='seismic') # plots of averages
#s.surf('vphi', r=0.9, cm = 'seismic') # surface plots 

gr.vtheta # theta component of velocity - numpy array
print(gr.vtheta.shape) # theta component of velocity - shape
gr.Btheta.shape # theta component of magnetic field - shape
gr.radius # radius  - numpy array
print(gr.radius.shape)  
gr.ntheta # number of pts in latitude
gr.nphi # number of pts in azimuth
# you have to "construct" the grid in phi and theta yourself - check the plotting routines 

#sp = MagicSpectrum(datadir=my_path,tag='rot01', field='e_mag', ispec=20) # 1D spectra
#sp = MagicSpectrum2D(datadir=my_path,tag='rot01', field='e_mag', ispec=20) # 2D spectra - throughs an error for now, check with MagIC website

ts = MagicTs(datadir=my_path,field='dipole',tag = 'rot01', all=True)
ts = MagicTs(datadir=my_path,field='par', tag = 'rot01', all=True)


