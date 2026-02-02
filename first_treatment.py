import numpy as np 
import matplotlib.pyplot as plt 
from magic import *
import glob #pour appeler tous les fichiers correspondants a un pattern : glob.glob('pattern')

my_path = '/travail/dynconv/multiscale_dyno/'
my_dir = 'Nrho4_Ek4'

# il y a 2 log un avec un tag thicka et l'autre thickb, raison ??
# datadir doit etre mis partout 
# ----------------------- log treatment ----------------------- 

stp = MagicSetup(datadir= my_path+my_dir,nml='log.thicka', quiet=True)
print(stp.ra) # print the Rayleigh number
print(stp.n_r_max) # print n_r_max

#--------------------- time serie treatment -------------------------
# e_kin, e_mag_oc, e_mag_ic, rot (rotation rates), dipole (dipolar components of the magnetic field), par (diagnostic parameter)
# + additionnal ones which are conditionnal : AM (angular momentum), heat, helicity, power and dtE (power budget), u_square (v**2), drift, SR[IC/MA] (torques), geos, dtVrms, dtBrms, perpPar (Kinetic energies perpendicular and parallel to the rotation axis)


# plot the most recent e_kin.TAG file found in the directoy
ts = MagicTs(datadir=my_path+my_dir,field='e_kin')

# stack **all** the dipole.TAG file found in the directory
ts1 = MagicTs(datadir=my_path+my_dir,field='dipole', all=True)
print(ts.time, ts.dipole) # print time and dipole

# If you only want to read the file ``heat.N0m2z``
ts2 = MagicTs(datadir=my_path+my_dir,field='heat', tag='thickb', iplot=False) #pas de plot

# Average only the files that match the pattern thick[a-b] from t=2.11
a = AvgField(datadir=my_path+my_dir,tstart=2.11, tag='thick[a-b]')
print(a) # print the formatted output

# Custom JSON model to select averages
json_model = { 'phys_params': ['ek'],
               'time_series': { 'heat': ['topnuss', 'botnuss'],
                                'e_kin': ['ekin_pol', 'ekin_tor'],
                                'par': ['rm'] },
                   'spectra': {},
                   'radial_profiles': {'powerR': ['viscDiss', 'buoPower']}
                 }

# Compute the selected averages in the dirctory mydir
a = AvgField(datadir=my_path+my_dir, model=json_model)

#----------------- treatment of time averaged radial profiles ----------------------
rad = MagicRadial(datadir=my_path+my_dir,field='eKinR') # display the content of eKinR.tag
print(rad.radius, rad.ekin_pol_axi) # print radius and poloidal energy

#----------------- spectra files ----------------------
# 1D
sp = MagicSpectrum(datadir=my_path+my_dir,field='e_kin', ispec=1) #first spectrum of ekin

# display the content of mag_spec_ave on one single figure
sp = MagicSpectrum(datadir=my_path+my_dir,field='e_mag', ave=True)

# display both kinetic and magnetic energy spectra on same graph
sp = MagicSpectrum(datadir=my_path+my_dir,field='combined', ave=True)

# 2D
# display the content of 2D_kin_spec_1 most recent file in the current directory with the colormap seismic and with 17 contours
sp = MagicSpectrum2D(datadir=my_path+my_dir, field='e_kin', ispec=1, levels=17, cm='seismic')

#----------------- GRAPH files ----------------------

# Regular G files

gr = MagicGraph(datadir=my_path+my_dir,ivar=1, tag='thicka')
print(gr.vr.shape) # shape of vr
print(gr.ek) # print ekman number
print(gr.minc) # azimuthal symmetry

# Averaged G file with double precision
gr = MagicGraph(datadir=my_path+my_dir,ave=True, tag='thicka', precision=np.float64)

# To read G_1.test
s = Surf(datadir=my_path+my_dir,ivar=1, ave=False, tag='thicka',precision=np.float64)

# ====== avg ======
# Axisymmetric zonal flows, 65 contour levels
s.avg(field='vp', levels=65, cm='seismic')
# Axisymmetric Bphi + poloidal field lines
s.avg(field='Bp', pol=True, polLevels=8)
# Omega-effect, contours truncated from -1e3 to 1e3
s.avg(field='omeffect', vmax=1e3, vmin=-1e3)
# + plein d'autres option qui peuvent changer l'affichage

# ===== equatorial cut ======
# Equatorial cut of the z-vorticity, 65 contour levels with the limit of the colormap from -1e3 to 1e3 and Normalise the contour levels radius by radius
s.equat(field='vortz', levels=65, cm='seismic', vmin=-1e3, vmax=1e3, normRad=True)

# ====== slice ======
# vphi at 0, 30, 60 degrees in longitude, contours truncated from -1e3 to 1e3
s.slice(field='vp', lon_0=[0, 30, 60], levels=65, cm='seismic', vmax=1e3, vmin=-1e3)
# Axisymmetric Bphi + poloidal field lines
s.avg(field='Bp', pol=True, polLevels=8)

# ===== surf ======
# Radial flow component at ``r=0.95 r_o``, 65 contour levels
s.surf(field='vr', r=0.95, levels=65, cm='seismic')
# If basemap is installed, additional projections are available
s.surf(field='Br', r=0.95, proj='ortho', lat_0=45, lon_0=45)


#----------------- MOVIE files ----------------------

# Read TO_mov.TAG, time-averaged it and display it with 65 contour levels
t = TOMovie(file='TO_mov.TAG', avg=True, levels=65, cm='seismic')

# read and display z-integrated quantities produced by the TO output
to = MagicTOHemi(hemi='n', iplot=True) # For the Northern hemisphere

