'''
defualt settings and parameters
'''
import numpy as np

class defaults:
    def __init__(self):
        self.set_survey_geometries()

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
            'sigma8_0': 0.9,
        }

        self.euclid = {
            'short_name': 'euclid',
            'name': 'Euclid',
            'f_sky': 15000/full_sky_deg,
            'ng_z_list': neuclid,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.9,
        }

        self.boss = {
            'short_name': 'boss',
            'name': 'BOSS',
            'f_sky': 10000/full_sky_deg,
            'ng_z_list': nboss,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.9,
        }

        self.roman = {
            'short_name': 'roman',
            'name': r"Roamn ($H\alpha$)",
            'f_sky': 2000/full_sky_deg, # https://www.stsci.edu/roman/about/science-themes
            'ng_z_list': nroman,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.9,
        }

        self.pfs = {
            'short_name': 'pfs',
            'name': 'PFS',
            'f_sky': 2000/full_sky_deg, # http://member.ipmu.jp/masahiro.takada/pfs_whitepaper.pdf
            'ng_z_list': npfs,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.9,
        }

        self.spherex = {
            'short_name': 'spherex',
            'name': 'Spherex',
            'f_sky': 0.75,
            'ng_z_list': nspherex,
            'Sigma_0': 16.6,
            'b_0': 0.95,
            'survey_type':'spectroscopic',
            'sigma8_0': 0.9,
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
            'sigma8_0': 0.9,
        }


