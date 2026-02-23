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

from magic import *

for a in ["gr/Nr2p5_Pm4/ra_8e6/om50","gr2/xi_p2_pm4/ra_1e6/om15","gr2/xi_p35_pm4/ra_5e6/om50"]:
	my_path = '/travail/dynconv/multiscale_dyno/anelasticCouette/'+a
	gr = MagicGraph(datadir=my_path,ivar=9,tag='rot01') # access to 3D fields 
	s = Surf(datadir=my_path,ivar=9, ave=False, tag='rot01')
	s.avg(field='vr', cm='seismic')
	s.avg(field='br', cm='seismic')
	s.avg(field='helicity', cm='seismic')
	
	thlin = np.linspace(0., np.pi, gr.ntheta)
	indices = np.where(thlin == np.pi*81/180)[0]
	filtre = (gr.vr[:,indices,:]).mean(axis=0)
	plt.figure(50)
	plt.plot(gr.radius,filtre)
plt.show()	

"""
s.avg(field='vtheta', cm='seismic')
s.avg(field='vphi', cm='seismic')
s.avg(field='btheta', cm='seismic')
s.avg(field='bphi', cm='seismic')


print(gr.__dict__.keys())
print(gr.vr.shape) # theta component of velocity - shape
print(gr.vtheta.shape)
print(gr.vphi.shape)
print(gr.Br.shape)  
print(gr.Btheta.shape) 
print(gr.Bphi.shape) 
print(gr.ntheta) # number of pts in latitude
print(gr.nphi) # number of pts in azimuth

s.surf('vr', r=0.9, cm = 'seismic')
s.surf('vtheta', r=0.9, cm = 'seismic')
s.surf('vphi', r=0.9, cm = 'seismic')
s.surf('br', r=0.9, cm = 'seismic')
s.surf('btheta', r=0.9, cm = 'seismic')
s.surf('bphi', r=0.9, cm = 'seismic')

s.equat(field='vr', levels=65, cm='seismic')
s.equat(field='vtheta', levels=65, cm='seismic')
s.equat(field='vphi', levels=65, cm='seismic')
s.equat(field='br', levels=65, cm='seismic')
s.equat(field='btheta', levels=65, cm='seismic')
s.equat(field='bphi', levels=65, cm='seismic')

s.avg(field='cr', cm='seismic')
"""

