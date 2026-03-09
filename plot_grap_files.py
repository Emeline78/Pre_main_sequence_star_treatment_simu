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
import glob
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

for a,xi in [("gr/Nr2p5_Pm4/ra_8e6/om50",0.2),("gr2/xi_p2_pm4/ra_1e6/om15",0.2),("gr2/xi_p35_pm4/ra_5e6/om50",0.35)]:
	my_path = '/travail/dynconv/multiscale_dyno/anelasticCouette/'+a
	gr = MagicGraph(datadir=my_path,ivar=9,tag='rot01') # access to 3D fields 
	#s = Surf(datadir=my_path,ivar=9, ave=False, tag='rot01')
	#s.avg(field='vr', cm='seismic')
	#s.avg(field='br', cm='seismic')
	#s.avg(field='helicity', cm='seismic')
	

paths = glob.glob("/travail/dynconv/multiscale_dyno/anelasticCouette/gr2/xi_p35_pm4/ra_5e6/om*")
n = len(paths)
for i,a in enumerate(paths) :
	gr = MagicGraph(datadir=a,tag='rot01')
	thlin = np.linspace(0., np.pi, gr.ntheta)
	indices = np.where(np.isclose(thlin, np.pi/2,atol=1e-2))[0]
	print(indices, thlin[indices])
	filtre = (gr.vr[:,indices,:]).mean(axis=0)
	rth = gr.radius * (1 - 0.35)
	color = plt.cm.Reds(0.3 + 0.7 * i / max(n-1, 1))
	plt.figure(1)
	plt.plot(rth,filtre[0],color = color)
plt.grid()
plt.show()

	


