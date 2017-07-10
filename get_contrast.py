import numpy as np
import itertools
import matplotlib.pyplot as plt
import csv
import cPickle as pickle
import numpy.polynomial.chebyshev as cheb 

def print_files():
    stars_str = np.genfromtxt('star_list_MOD.dat',dtype=str)
    stars_val = np.genfromtxt('star_list_MOD.dat',dtype=str)

    G2_val = np.genfromtxt('summary_sel_G2.rdb')
    G2_str = np.genfromtxt('summary_sel_G2.rdb',dtype=str)

    K5_val = np.genfromtxt('summary_sel_K5.rdb')
    K5_str = np.genfromtxt('summary_sel_K5.rdb',dtype=str)

    M2_val = np.genfromtxt('summary_sel_M2.rdb')
    M2_str = np.genfromtxt('summary_sel_M2.rdb',dtype=str)

    G2_file = open('G2_match.dat','w')
    K5_file = open('K5_match.dat','w')
    M2_file = open('M2_match.dat','w')


    n_stars = np.size(stars_str[:,0])
    for ii in xrange(0,n_stars):
        name, snr, teff, logg, vmic, gfeh =  stars_str[ii,:]
        ind = np.where((G2_str[:,0]==name) & (G2_val[:,1]>10))
        if np.size(ind)>0:
            #print name, teff, logg, vmic, gfeh, np.median(G2_val[ind,2]), np.median(G2_val[ind,3])
            G2_file.write(
                        '{0:10s} {1:10s} {2:8s} {3:8s} {4:8s} {5:8.4f} {6:7.2f} \n'.format(
                        name, teff, logg, vmic, gfeh, np.median(G2_val[ind,2]), np.median(G2_val[ind,3])))

        ind = np.where((K5_str[:,0]==name) & (K5_val[:,1]>10))
        if np.size(ind)>0:
            #print name, teff, logg, vmic, gfeh, np.median(G2_val[ind,2]), np.median(G2_val[ind,3])
            K5_file.write(
                        '{0:10s} {1:10s} {2:8s} {3:8s} {4:8s} {5:8.4f} {6:7.2f} \n'.format(
                        name, teff, logg, vmic, gfeh, np.median(K5_val[ind,2]), np.median(K5_val[ind,3])))

        ind = np.where((M2_str[:,0]==name) & (M2_val[:,1]>10))
        if np.size(ind)>0:
            #print name, teff, logg, vmic, gfeh, np.median(G2_val[ind,2]), np.median(G2_val[ind,3])
            M2_file.write(
                        '{0:10s} {1:10s} {2:8s} {3:8s} {4:8s} {5:8.4f} {6:7.2f} \n'.format(
                        name, teff, logg, vmic, gfeh, np.median(M2_val[ind,2]), np.median(M2_val[ind,3])))

    G2_file.close()
    K5_file.close()
    M2_file.close()


def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def do_contrast_cal():
    labels = ['G2','K5']

    for lab in labels:
        data = np.genfromtxt(lab + '_match.dat')
        sel = (data[:,5] < 9)
        teff, logg, vmic, gfeh, fwhm, contrast = data[sel,1:].T

        # Fit a 3rd order, 2d polynomial
        m = polyfit2d(teff,gfeh,contrast)

        z = polyval2d(teff, gfeh, m)
        cond_sel = (np.abs(contrast-z) < 3.)
        m = polyfit2d(teff[cond_sel],gfeh[cond_sel],contrast[cond_sel])

        print 'Average difference:  ', np.average(contrast-z)
        print 'Standard deviation:  ', np.std(contrast-z)
        print

        pickle.dump(m, open(lab + '_contrast_2Dcoeff.pick', 'wb'))

        # Evaluate it on a grid...
        if lab =='G2':
            nx, ny = 21, 19
            xx, yy = np.meshgrid(np.linspace(4500, 6500, nx),
                             np.linspace(-1.3, 0.5, ny))

        if lab =='K5':
            nx, ny = 14, 19
            xx, yy = np.meshgrid(np.linspace(4500, 5800, nx),
                         np.linspace(-1.3, 0.5, ny))

        zz = polyval2d(xx, yy, m)
        y=yy[:,0]
        x=xx[0,:]


        h2d_out = np.zeros([ny+1, nx+1])
        h2d_out[0, 1:] = x
        h2d_out[1:, 0] = y
        h2d_out[1:, 1:] = zz
        h2d_list =  h2d_out.tolist()
        h2d_list[0][0] = ''
        csvfile = lab+'_contrast_surface.csv'
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(h2d_list)

def do_FWHM_cal():
        data = np.genfromtxt('G2K5_match.dat')
        sel = (data[:,5] > 5) & (data[:,5] < 12)
        teff, logg, vmic, gfeh, fwhm, contrast = data[sel,1:].T

        # m = cheb.chebfit(teff[sel],fwhm[sel],5,full=True)
	# z = cheb.chebval(teff,m[0]) 
        # cond_sel = (np.abs(fwhm-z) < 0.5)
        # m = cheb.chebfit(teff[cond_sel],fwhm[cond_sel],5,full=True)
	# x_plt = np.arange(4000,7000,50)
	# y_plt = cheb.chebval(x_plt,m[0]) 
	
        m = np.polyfit(teff,fwhm,3)
	z = np.polyval(m,teff)
        cond_sel = (np.abs(fwhm-z) < 0.5)        
        m = np.polyfit(teff[cond_sel],fwhm[cond_sel],3)
	x_plt = np.arange(4000,7000,50)
	y_plt = np.polyval(m,x_plt) 

        print 'Average difference:  ', np.average(fwhm-z)
        print 'Standard deviation:  ', np.std(fwhm-z)
        print

        pickle.dump(m, open('G2K5_fwhm_1Dcoeff.pick', 'wb'))
	
	fileout = open('G2K5_fwhm_fit.dat','w')
        for ii in xrange(0,np.size(x_plt)):
            fileout.write('{0:8.4f} {1:8.4f}  \n'.format(x_plt[ii], y_plt[ii]))
	fileout.close()



input_pams = np.genfromtxt('K2_list.dat',delimiter=',', missing_values=9999.0)
#input_name = np.genfromtxt('K2_list.dat',delimiter=',', missing_values='9999')

G2_contrast_coeff = pickle.load(open('G2_contrast_2Dcoeff.pick', 'rb'))
K5_contrast_coeff = pickle.load(open('K5_contrast_2Dcoeff.pick', 'rb'))
G2K5_fwhm_coeff = pickle.load(open('G2K5_fwhm_1Dcoeff.pick', 'rb'))


K2name, Vmag, teff, gfeh = input_pams.T  
     
G2_sel = (teff > 5300.00) & (teff <  7000)
K5_sel = (teff > 4500.00) & (teff <= 5300)
M2_sel = (teff > 2000.00) & (teff <= 4500)
AA_sel = (teff > 2000.00) & (teff <  7000)

teff[M2_sel] = 4500
gfeh[M2_sel] = 0.00

contrast = np.zeros(np.size(teff))
fwhm =  np.zeros(np.size(teff))
output = np.zeros([np.size(teff),2])

contrast[G2_sel] = polyval2d(teff[G2_sel], gfeh[G2_sel], G2_contrast_coeff)
contrast[K5_sel] = polyval2d(teff[K5_sel], gfeh[K5_sel], K5_contrast_coeff)
contrast[M2_sel] = polyval2d(teff[M2_sel], gfeh[M2_sel], K5_contrast_coeff)

fwhm[AA_sel] = np.polyval(G2K5_fwhm_coeff,teff[AA_sel])
output[:,0] = contrast
output[:,1] = fwhm
fileout = open('cal_out.dat','w')

for pams in output:
    fileout.write('{0:8.4f} {1:8.4f}  \n'.format(pams[0], pams[1]))
fileout.close()