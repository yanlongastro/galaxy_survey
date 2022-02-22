'''
defualt settings and parameters
'''
import numpy as np
import fisher_matrix as fm

class defaults:
    def __init__(self):
        self.set_survey_geometries()
        self.set_alpha_prior()
        

    def set_alpha_prior(self):
        sf_p18 = 97.288
        dsf_p18 = 0.799344
        sa2 = 1./(dsf_p18/sf_p18)**2
        sb2 = 1e-20
        matrix = np.diag([sa2, sb2])
        keys = ['alpha', 'beta']
        fs = fm.fisher(matrix, keys)
        self.alpha_prior = fs

    def set_survey_geometries(self):
        full_sky_deg = 4*np.pi*(180/np.pi)**2
        self.full_sky_deg = full_sky_deg

        ndesi = np.loadtxt('desi-density.txt')
        neuclid = np.loadtxt('euclid-density.txt')
        nboss = np.loadtxt('boss-density.txt')
        ndesi[:,1]/=1e3
        neuclid[:,1]/=1e3
        nboss[:,1]/=1e3

        nroman = np.loadtxt('roman-ha-density.txt')
        npfs = np.loadtxt('pfs-density.txt')
        nspherex = np.loadtxt('spherex-density.txt')

        self.desi = {
            'short_name': 'desi',
            'name': 'DESI',
            'f_sky': 14000/full_sky_deg,
            'ng_z_list': ndesi,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.811,  #planck 18
        }

        self.euclid = {
            'short_name': 'euclid',
            'name': 'Euclid',
            'f_sky': 15000/full_sky_deg,
            'ng_z_list': neuclid,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.811,
        }

        self.boss = {
            'short_name': 'boss',
            'name': 'BOSS',
            'f_sky': 10000/full_sky_deg,
            'ng_z_list': nboss,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.811,
        }

        self.roman = {
            'short_name': 'roman',
            'name': r"Roman (H$\alpha$)",
            'f_sky': 2000/full_sky_deg, # https://www.stsci.edu/roman/about/science-themes
            'ng_z_list': nroman,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.811,
        }

        self.pfs = {
            'short_name': 'pfs',
            'name': 'PFS',
            'f_sky': 2000/full_sky_deg, # http://member.ipmu.jp/masahiro.takada/pfs_whitepaper.pdf
            'ng_z_list': npfs,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.811,
        }

        self.spherex = {
            'short_name': 'spherex',
            'name': 'SPHEREx',
            'f_sky': 0.75,
            'ng_z_list': nspherex,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.811,
        }

        self.cvl = {
            'short_name': 'cvl',
            'name': 'CVL',
            'f_sky': 0.5,
            'N_g': 1e100,
            'z_min': 0.1,
            'z_max': 4.0,
            'dz': 0.1,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.811,
        }


anu = 8/7*(11/4)**(4/3)
nnu = 3.046
amp = (anu+nnu)/nnu

def sb2sn(x):
    return 1/0.194057*x

def sn2sb(y):
    return 0.194057*y

def sb2sn_full(x, b=1):
    return anu/(amp-b) - anu* (-x + b)/(amp - (-x + b))

def sn2sb_full(y, n=3.046):
    b = n2b(n)
    t = anu/(amp-b) -y
    return b - amp*t/(t+anu)

def b2n(x):
    return anu* (x)/(amp - (x))
    
def n2b(y):
    return amp*y/(y+anu)