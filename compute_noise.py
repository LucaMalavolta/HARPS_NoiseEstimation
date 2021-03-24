import numpy as np
import itertools
import matplotlib.pyplot as plt
import csv
import pickle
import numpy.polynomial.chebyshev as cheb
import routines.common as cm

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

input_pams = np.atleast_2d(np.genfromtxt('input.list',delimiter=',', missing_values=9999.0, names=True))
star_name = np.genfromtxt('input.list',delimiter=',', missing_values=9999.0, skip_header=1, dtype=str, usecols=0)

G2_contrast_coeff = pickle.load(open(cm.base_dir + 'G2_contrast_2Dcoeff.pick', 'rb'))
K5_contrast_coeff = pickle.load(open(cm.base_dir + 'K5_contrast_2Dcoeff.pick', 'rb'))
G2K5_fwhm_coeff = pickle.load(open(cm.base_dir + 'G2K5_fwhm_1Dcoeff.pick', 'rb'))

teff = input_pams['Teff'][0]
Vmag = input_pams['Vmag'][0]
gfeh = input_pams["gfeh"][0]
vsini = input_pams['vsini'][0]

G2_sel = (teff > 5300.00) & (teff <  7000.00)
K5_sel = (teff > 4500.00) & (teff <= 5300.00)
M2_sel = (teff > 2000.00) & (teff <= 4500.00)
AA_sel = (teff > 2000.00) & (teff <  7000.00)

teff[M2_sel] = 4500.00
gfeh[M2_sel] = 0.00

contrast = np.zeros(np.size(teff))
fwhm =  np.zeros(np.size(teff))
output = np.zeros([np.size(teff),2])

contrast[G2_sel] = polyval2d(teff[G2_sel], gfeh[G2_sel], G2_contrast_coeff)
contrast[K5_sel] = polyval2d(teff[K5_sel], gfeh[K5_sel], K5_contrast_coeff)
contrast[M2_sel] = polyval2d(teff[M2_sel], gfeh[M2_sel], K5_contrast_coeff)

fwhm[AA_sel] = np.polyval(G2K5_fwhm_coeff,teff[AA_sel])


#rv_noise = np.sqrt(np.sqrt(fwhm**2+vsini**2)/6.5)/(contrast/45.0)/(1.17*(39.0*np.sqrt(2.0)*10**(-0.2*(Vmag-12.0)))/91.00)*100

# seeing correction
#f = 1.17 - 0.70 (theta - 1.24)
f = 1 # we set this factor to one because we are not modelling the seeing effect

# SNR as a function of exposure time:
# s = 39 * np.sqrt( t / 1800 ) * 10 **(-0.2*(V-12))
s = 36. * np.sqrt( 3600. / 1800. ) * 10 **(-0.2*(Vmag-12)) # rescaled factor to take into account the real efficiency of HARPS-N

np.seterr(divide='ignore', invalid='ignore')

fwhm_observed = np.sqrt(fwhm**2+vsini**2)
contrast_observed = contrast*fwhm / fwhm_observed

rv_noise = np.sqrt(fwhm_observed/6.5)/ (f*s / 91.00)
total_noise = np.sqrt(0.4**2 + rv_noise**2)* 100.

fileout = open('cal_out.dat','w')
fileout.write('Name, Vmag, Teff, gfeh, vsini, fwhm, contrast, fwhm_observed, contrast_observed, rv_noise\n')
for s,v,t,g,vs,f,c,fo,co,r in zip(star_name, Vmag, teff, gfeh, vsini, fwhm, contrast, fwhm_observed, contrast_observed, total_noise):
    fileout.write('{0:s}, {1:8.2f}, {2:8.2f}, {3:8.2f}, {4:8.2f}, {5:8.2f}, {6:8.2f}, {7:8.2f}, {8:8.2f}, {9:8.2f}\n'.format(s,v,t,g,vs,f,c,fo,co,r))
fileout.close()
