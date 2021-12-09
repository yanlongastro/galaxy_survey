from math import cos
import numpy as np
from tqdm.notebook import tqdm
import camb
import copy
from functools import lru_cache

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams["figure.dpi"] = 100

import dewiggle as dw
import galaxy_survey as gs
import defaults as df

class camb_cosmology:
    def __init__(self, parameters=None, fiducial_parameters=None, fix_H0=False):
        '''
        parameters must use standard camb varaiable names
        '''
        self.parameters = self.set_parameters(parameters)
        self.camb = self.run_camb(self.parameters, fix_H0=fix_H0)
        if fiducial_parameters is None:
            self.fiducial_parameters = self.parameters
            self.fiducial_camb = self.camb
        else:
            self.fiducial_parameters = self.set_parameters(fiducial_parameters)
            self.fiducial_camb = self.run_camb(self.fiducial_parameters, fix_H0=fix_H0)
        self.get_ap_factors()
        self.get_power_spectrum()
        self.h = self.camb.hubble_parameter(0.)/100
        self.rstar = self.camb.get_derived_params()['rstar']*self.h
        self.fix_H0 = fix_H0

        self.alpha = self.fiducial_camb.get_derived_params()['rstar']*self.fiducial_camb.hubble_parameter(0.)/self.camb.get_derived_params()['rstar']/self.camb.hubble_parameter(0.)
        self.beta = df.n2b(self.parameters['nnu']['value'])


    def set_parameters(self, parameters):
        default_parameters = {'ombh2': {'value': 0.02237, 'stdev': 0.00015, 'h': 0.0008},
                            'omch2': {'value': 0.1200, 'stdev': 0.0012, 'h': 0.001},
                            'As': {'value': 2.1413038238853928e-09,
                            'stdev': 2.959130012032031e-11,
                            'h': 1.0000000000000002e-10},
                            'ns': {'value': 0.9649, 'stdev': 0.0042, 'h': 0.01},
                            'tau': {'value': 0.0544, 'stdev': 0.0073, 'h': 0.004},
                            'YHe': {'value': 0.2478, 'stdev': 0.025, 'h': 0.005},
                            'thetastar': {'value': 0.0104112, 'stdev': 3.1e-06, 'h': 1e-05},
                            'nnu': {'value': 3.046, 'stdev': 1e+100, 'h': 0.01},
                            'H0':{'value': 67.4},
                            }
        
        # default_parameters = {'ombh2': {'value': 0.0223, 'stdev': 0.00015, 'h': 0.0008},
        #                     'omch2': {'value': 0.1188, 'stdev': 0.0012, 'h': 0.002},
        #                     'As': {'value': 2.1413038238853928e-09,
        #                     'stdev': 2.959130012032031e-11,
        #                     'h': 1.0000000000000002e-10},
        #                     'ns': {'value': 0.9667, 'stdev': 0.0042, 'h': 0.01},
        #                     'tau': {'value': 0.066, 'stdev': 0.0073, 'h': 0.02},
        #                     'YHe': {'value': 0.2478, 'stdev': 0.025, 'h': 0.005},
        #                     'thetastar': {'value': 0.0104112, 'stdev': 3.1e-06, 'h': 2e-5},
        #                     'nnu': {'value': 3.046, 'stdev': 1e+100, 'h': 0.08},
        #                     'H0':{'value': 67.4},
        #                     }

        self.default_parameters = default_parameters
        if parameters is not None:
            for k in parameters.keys():
                if k in default_parameters.keys():
                    default_parameters[k]['value'] = parameters[k]['value']
                    for kk in ['stdev', 'h']:
                        if kk in parameters[k].keys():
                            default_parameters[k][kk] = parameters[k][kk]
        return default_parameters


    def run_camb(self, parameters=None, fix_H0=False):
        if parameters is None:
            parameters = self.default_parameters
        pars = camb.CAMBparams()
        if fix_H0:
            H0 = parameters['H0']['value']
            thetastar = None
        else:
            thetastar = parameters['thetastar']['value']
            H0 = None
        pars.set_cosmology(H0=H0, 
                            ombh2=parameters['ombh2']['value'], 
                            omch2=parameters['omch2']['value'], 
                            tau=parameters['tau']['value'], 
                            YHe=parameters['YHe']['value'], 
                            thetastar=thetastar, 
                            nnu=parameters['nnu']['value'],
                            )
        pars.InitPower.set_params(As=parameters['As']['value'], ns=parameters['ns']['value'])
        pars.set_matter_power(kmax=10)
        pars.NonLinear = camb.model.NonLinear_none
        results = camb.get_results(pars)
        return results


    def get_ap_factors(self, z_interp_array=None):
        h_hfid = self.camb.hubble_parameter(0.)/self.fiducial_camb.hubble_parameter(0.)
        # we used Mpc/h so AP has no effect at z=0.
        q_parallel = lambda z: self.fiducial_camb.hubble_parameter(z)/self.camb.hubble_parameter(z)*h_hfid
        q_vertical = lambda z: self.camb.angular_diameter_distance(z)/self.fiducial_camb.angular_diameter_distance(z)*h_hfid
        # q_parallel != q_vertical unless z=0.
        q_isotropic = lambda z: pow(q_vertical(z), 2./3.)*pow(q_parallel(z), 1./3.)

        if z_interp_array is None:
            z_interp_array = np.linspace(1e-5, 10., num=401)
        nz = len(z_interp_array)
        qp = q_parallel(z_interp_array)
        qv = q_vertical(z_interp_array)
        qr = q_isotropic(z_interp_array)
        q_parallel = lru_cache(maxsize=None)(lambda x: gs.two_value_interpolation_c(z_interp_array, qp, x, nz))
        q_vertical = lru_cache(maxsize=None)(lambda x: gs.two_value_interpolation_c(z_interp_array, qv, x, nz))
        q_isotropic = lru_cache(maxsize=None)(lambda x: gs.two_value_interpolation_c(z_interp_array, qr, x, nz))
        self.q_parallel = q_parallel
        self.q_vertical = q_vertical
        self.q_isotropic = q_isotropic
        return q_parallel, q_vertical, q_isotropic

    def get_power_spectrum(self, num_k=1e6):
        n = int(num_k)
        kh = np.logspace(-4, 0.5, num=n)
        pk = self.camb.get_matter_power_interpolator(nonlinear=False).P(0., kh)
        matter_power_spectrum, matter_power_spectrum_no_wiggle = dw.dewiggle(kh, pk)

        kh = np.linspace(1e-4, 1, num=n)
        pk = np.array([matter_power_spectrum(x) for x in kh])
        pknw = np.array([matter_power_spectrum_no_wiggle(x) for x in kh])
        self.matter_power_spectrum = lru_cache(maxsize=None)(lambda x: gs.two_value_interpolation_c(kh, pk, x, n))
        self.matter_power_spectrum_no_wiggle = lru_cache(maxsize=None)(lambda x: gs.two_value_interpolation_c(kh, pknw, x, n))

        kh = np.linspace(1e-4, 1, num=n)
        osc = np.array([self.matter_power_spectrum(x)/self.matter_power_spectrum_no_wiggle(x)-1. for x in kh])
        self.oscillation_part = lru_cache(maxsize=None)(lambda x: gs.two_value_interpolation_c(kh, osc, x, n))
        dodk = np.diff(osc)/np.diff(kh)
        self.oscillation_part_derivative = lru_cache(maxsize=None)(lambda x: gs.two_value_interpolation_c(kh[:-1], dodk, x, n-1))


    def power_spectrum_derivative_parts_factory(self, parameters_plus, parameters_minus, h):
        '''
        returns: power spectrum, oscillation part without ap, ap_factors, stepsize
        '''
        camb_plus = camb_cosmology(parameters_plus, fiducial_parameters=self.parameters)
        camb_minus = camb_cosmology(parameters_minus, fiducial_parameters=self.parameters)
        plus = {}
        plus['matter_power_spectrum'] = camb_plus.matter_power_spectrum
        plus['matter_power_spectrum_no_wiggle'] = camb_plus.matter_power_spectrum_no_wiggle
        plus['oscillation_part'] = camb_plus.oscillation_part
        plus['q_parallel'] = camb_plus.q_parallel
        plus['q_vertical'] = camb_plus.q_vertical
        plus['q_isotropic'] = camb_plus.q_isotropic

        minus = {}
        minus['matter_power_spectrum'] = camb_minus.matter_power_spectrum
        minus['matter_power_spectrum_no_wiggle'] = camb_minus.matter_power_spectrum_no_wiggle
        minus['oscillation_part'] = camb_minus.oscillation_part
        minus['q_parallel'] = camb_minus.q_parallel
        minus['q_vertical'] = camb_minus.q_vertical
        minus['q_isotropic'] = camb_minus.q_isotropic

        res = {'plus': plus,
                'minus': minus,
                'h': h}
        return res

    def prepare_power_spectrum_derivative_parts_single(self, key):
        pp = copy.deepcopy(self.parameters)
        pp[key]['value'] += pp[key]['h']

        pm = copy.deepcopy(self.parameters)
        pm[key]['value'] -= pm[key]['h']

        h = pp[key]['h']
        return self.power_spectrum_derivative_parts_factory(pp, pm, h)

    def prepare_power_spectrum_derivative_parts(self, keys):
        res = {}
        for key in keys:
            if self.fix_H0 and key=='thetastar':
                continue
            if not self.fix_H0 and key=='H0':
                continue
            res[key] = self.prepare_power_spectrum_derivative_parts_single(key)
        self.power_spectrum_derivative_parts = res


class galaxy_correlations:
    '''
    Might not proceed with this one, since the speed penalty is quite siginificant
    '''
    def __init__(self, cosmology, fiducial_cosmology=None):
        '''
        cosmology/fiducial_cosmology: camb_cosmology, or ps_interpolation (check this)
        '''
        if fiducial_cosmology is None:
            self.fiducial_cosmology = cosmology
        self.cosmology = cosmology


    def galaxy_power_spectrum(k, mu=0, z=0, broadband_from_fiducial=False):
        2333


    def galaxy_bispectrum(broadband_from_fiducial=False):
        2333


    def galaxy_power_spectrum_derivative(self, cosmology_plus, cosmology_minus, h, wiggle_only=False):
        '''
        cosmology_plus/minus: two set of cosmology by adding/substracting the stepsize.
        '''
        gc_plus = galaxy_correlations(cosmology_plus, fiducial_cosmology=self.cosmology)
        gc_minus = galaxy_correlations(cosmology_minus, fiducial_cosmology=self.cosmology)
        derivative = lambda k, mu=0., z=0.: (gc_plus.galaxy_power_spectrum(k, mu, z, broadband_from_fiducial=wiggle_only) - gc_minus.galaxy_power_spectrum(k, mu, z, broadband_from_fiducial=wiggle_only))/2./h
        return derivative        