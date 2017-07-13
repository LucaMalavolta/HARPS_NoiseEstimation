import numpy as np
import routines.kepler_exo as kp
from routines.get_mass import get_mass_from_K

import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


period_range = 10**np.arange(np.log10(0.5), np.log10(1000), 0.01)
mass_range = 10**np.arange(np.log10(1.), np.log10(800), 0.05)
mass_range_solar = mass_range / 330000

M_star1 = 1.4000

input_pams = np.atleast_2d(np.genfromtxt('input.list',delimiter=',', missing_values=9999.0, names=True))
noise_pams = np.atleast_2d(np.genfromtxt('cal_out.dat',delimiter=',', missing_values=9999.0, names=True))
star_name = np.genfromtxt('input.list',delimiter=',', missing_values=9999.0, skip_header=1, dtype=str, usecols=0)
star_check = np.genfromtxt('cal_out.dat',delimiter=',', missing_values=9999.0, skip_header=1, dtype=str, usecols=0)

Mstar = input_pams['Mstar'][0]
Rstar = input_pams['Rstar'][0]

rv_noise = noise_pams['rv_noise'][0]/100.
inclination = input_pams['planet_i'][0]

for n_star, this_star in enumerate(star_name):

    if star_check[n_star]==this_star:
        n_check= n_star
    else:
        n_check = np.where(star_check==this_star)[0][0]

    K_error = rv_noise[n_check]
    M_star1 = Mstar[n_star]
    R_star1 = Rstar[n_star]
    planet_i = inclination[n_star]


    K = np.zeros([len(mass_range),len(period_range)])
    sigma3 = np.zeros([len(mass_range),len(period_range)])
    for n_M, M_star2 in enumerate(mass_range_solar):
            K[n_M,:]= kp.kepler_K1(M_star1, M_star2, period_range, 90.000, 0.000)
            sigma3[n_M,:] = 1. / kp.kepler_K1(M_star1, M_star2, period_range, 90.000, 0.000)

    Period, Mass = np.meshgrid(period_range, mass_range)
    V=2**(np.arange(np.log2(0.5),np.log2(np.amax(K)),1))


    period_bounds = [period_range.min(), period_range.max() ]
    M_bounds = np.zeros(len(period_bounds))
    M_50N = np.zeros(len(period_bounds))
    M_jit = np.zeros(len(period_bounds))
    for n_P, P in enumerate(period_bounds):
        M_bounds[n_P] = get_mass_from_K(M_star1, P, K_error, 0.0000)
        M_50N[n_P] = get_mass_from_K(M_star1, P, np.sqrt((K_error**2+1.44)/(50./2.)), 0.0000)
        M_jit[n_P] = get_mass_from_K(M_star1, P, 1.2, 0.0000)

    """Compute transit geometry - minimum period allowed for non-transiting planet when assuming
    co-planary, 5deg tilting (best case scenario) and 10deg tilting (best case scenario)"""
    if planet_i > 89.90: planet_i = 89.90
    cop_P = np.sqrt( (R_star1/np.cos(planet_i*np.pi/180.)*6.957)**3 *(4*np.pi**2)/(1.98850 * 6.67408)) / 8.640
    d05_P = np.sqrt( (R_star1/np.cos((planet_i-5.00)*np.pi/180.)*6.957)**3 *(4*np.pi**2)/(1.98850 * 6.67408)) / 8.640
    d10_P = np.sqrt( (R_star1/np.cos((planet_i-10.00)*np.pi/180.)*6.957)**3 *(4*np.pi**2)/(1.98850 * 6.67408)) / 8.640

    """Compute transit depth, lazily adapted from
    http://phl.upr.edu/library/notes/standardmass-radiusrelationforexoplanets
    Not very accurate but good enough for our goals"""
    depth = 0.0002 # depth limit for Kepler detection
    Rplanet = np.sqrt(depth)*R_star1*109.2
    ###AAAA
    if Rplanet < 14.142135623730951:
        Mlimit = Rplanet**2
    else:
        Mlimit = (Rplanet/22.6)**-11.302083116219778


    fig0 = plt.figure(0, figsize=(10, 10))
    plt.figure(0)
    plt.xlabel('Period [d]')
    plt.ylabel('Planet Mass [M]')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([period_range.min(), period_range.max() ])
    plt.ylim([mass_range.min(), mass_range.max() ])

    plt.plot(period_bounds,M_bounds, c='k', label='CCF noise limit', lw=2)
    plt.plot(period_bounds,M_jit, c='r', label='Instrumental stability limit', lw=2, linestyle='dotted')
    plt.plot(period_bounds,M_50N, c='b', label='50-observations limit', lw=2, linestyle='dashed')

    plt.axhline(Mlimit, c='g', label='Lower limit on Mass for transit detection')

    plt.axvline(cop_P, c='c', label='Upper limit on Period for co-planar transiting planets')
    plt.axvline(d05_P, c='m', label='Upper limit on Period for i+5 deg transiting planets', linestyle='dotted')
    plt.axvline(d10_P, c='y', label='Upper limit on Period for i+10 deg transiting planets',  linestyle='dashed')
    CS = plt.contour(Period, Mass, K, V, label='Expected RV semi-amplitude')
    plt.clabel(CS, inline=1, fontsize=8)
    plt.title(this_star + '  Expected semi-amplitude')
    plt.legend()
    plt.savefig(this_star + '_RVdetectability.pdf', bbox_inches='tight', dpi=300)
    plt.close(0)
