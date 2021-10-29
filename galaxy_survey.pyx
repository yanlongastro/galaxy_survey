# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:03:10 2020

Cosmological Galaxy Survey (Cython version)

This is the updated version of the old 'galsurveycy', we drop the interpolated power spectrum and use the one calculated with camb instead.
Structure of the code changed accordingly.

@author: yanlong@caltech.edu
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
#from astropy.cosmology import Planck15
#from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import misc
from scipy import integrate
from scipy import stats
import functools
import itertools
import pathos.pools as pp
from multiprocessing import cpu_count, Pool
import time
import sobol_seq


cimport cython
cimport numpy as np
from libc.math cimport log, sqrt, log10, pi, INFINITY, pow, abs, exp
from cython.parallel import prange
DTYPE = np.float64

import fisher_matrix as fm
import galaxy_correlations as gc
import defaults as df


def f_phase(double k): 
    return 0.227*pow(k, 0.872)/(pow(k, 0.872)+pow(0.0324, 0.872))

def k_tf(double k1, double delta, double theta):
    return k1, k1+delta, sqrt(pow(k1, 2)+pow(k1+delta, 2)+2*k1*(k1+delta)*np.cos(theta))

def k_tf_as(double k1, double delta1, double delta2):
    return k1, k1+delta1, k1+delta1+delta2


def mu_tf(double mu_r, double xi, double cos12):
    mu1_t = mu_r*np.cos(xi)
    mu2_t = mu_r*np.sin(xi)
    mu1_t *= sqrt(2*(1-cos12))
    mu2_t *= sqrt(2*(1+cos12))
    mu1 = (mu1_t+mu2_t)/2
    mu2 = (-mu1_t+mu2_t)/2
    return mu1, mu2


def beta(double x):
    if x==1. or x==-1.:
        return .5
    elif -1.<x<1.:
        return 1
    else:
        return 0
#beta = np.vectorize(beta)


def is_zero(x):
    if x ==0:
        return 0.0
    else:
        return 1.0
#is_zero = np.vectorize(is_zero)

def is_unique(double k1, double k2, double k3):
    """
    This is actually a region function
    """
    return k1<=k2 and k2<=k3
#is_unique = np.vectorize(is_unique)

def cost(double k1, double k2, double k3):
    return (k3*k3-k1*k1-k2*k2)/(2*k1*k2)

def f_kernal(double k1, double k2, double cos12):
    return 5./7.+.5*(k1/k2+k2/k1)*cos12+2./7.*cos12*cos12

def g_kernal(double k1, double k2, double cos12):
    return 3./7.+.5*(k1/k2+k2/k1)*cos12+4./7.*cos12*cos12

def sigma_angle(double mu1, double mu2, double cos12):
    cdef double res = 1 - cos12*cos12 -mu1*mu1 - mu2*mu2 + 2*mu1*mu2*cos12
    if res <= 0.:
        return 0.0
    res = 1/(2*pi* sqrt(res))
    return res

#sigma_angle = np.vectorize(sigma_angle)

def s123(double k1, double k2, double k3):
    if k1==k2==k3:
        return 6
    elif k1==k2 or k2==k3 or k3==k1:
        return 2
    else:
        return 1




@cython.boundscheck(False)
@cython.wraparound(False)
def two_value_interpolation_c(np.ndarray[np.float64_t] x, np.ndarray[np.float64_t] y, np.float64_t val, np.int64_t n):
    '''
    Input must be equally gapped
    '''
    cdef  int index, index_max, index_min, index_mid
    cdef long double _xrange, xdiff, modolo, ydiff
    cdef long double y_interp

    # index = 0
    # while x[index] <= val:
    #    index += 1

    if val<=x[0]:
        return y[0]
    if val>=x[n-1]:
        return y[n-1]

    index_max = n
    index_min = 0
    while index_min < index_max:
       index_mid = index_min + ((index_max-index_min)>>1)
       if x[index_mid] <= val:
           index_min = index_mid+1
       else:
           index_max = index_mid
    index = index_min

    _xrange = x[index] - x[index-1]
    xdiff = val - x[index-1]
    modolo = xdiff/_xrange
    ydiff = y[index] - y[index-1]
    y_interp = y[index-1] + modolo*ydiff
    return y_interp
    

class cosmology:
    """
    cosmology parameters: Omega_m, Omega_b, Omega_L, h, s_f, sigma8_0
    """
    def __init__(self, cosmology_parameters):
        for key in cosmology_parameters:
            setattr(self, key, cosmology_parameters[key])

        # self.Omega_m = cosmology_parameters['Omega_m']
        # self.Omega_b = cosmology_parameters['Omega_b']
        # self.Omega_L = cosmology_parameters['Omega_L']
        # self.h = cosmology_parameters['h']
        # self.s_f = cosmology_parameters['s_f']
        self.astropy_cosmology = FlatLambdaCDM(H0=self.h*100.0, Om0=self.Omega_m)
        self.D0 = self.linear_growth_factor(0)

    def linear_growth_rate(self, np.float64_t z):
        cdef np.float64_t Om = self.Omega_m
        cdef np.float64_t Ol = self.Omega_L
        cdef np.float64_t f_growth = pow(Om*pow(1+z, 3)/(Om*(1+z)-(Om+Ol-1)*pow(1+z, 2)+Ol), 4./7.)
        return f_growth

    def linear_growth_factor(self, np.float64_t z):
        cdef np.float64_t Om = self.Omega_m
        cdef np.float64_t Ol = self.Omega_L
        cdef np.float64_t E = sqrt(Om*pow(1+z, 3) +(1-Om-Ol)*pow(1+z, 2) +Ol)
        cdef np.float64_t Omega_z = Om*pow(1+z, 3)/pow(E, 2)
        cdef np.float64_t lambda_z = Ol/pow(E, 2)
        cdef np.float64_t D_growth = 2.5/(1+z)*Omega_z/(pow(Omega_z, 4./7.)-lambda_z+(1+Omega_z/2)*(1+lambda_z/70))
        return D_growth

    def comoving_distance(self, z):
        """
        unit: Mpc/h
        """
        return(self.astropy_cosmology.comoving_distance(z).value*self.h)

    def sigma_8(self, z):
        return self.sigma8_0*self.linear_growth_factor(z)/self.D0

            

class survey:
    """
    cosmological_parameters (dict): refer camb_cosmology for details. Must be compatible.
    survey_geometrics (dict): f_sky, N_g, z_min, z_max, dz, ng_z_list ([zmid_list, ng_list]), Sigma_0, reconstruction_rate, b_0, survey_type, sigma_p, sigma8_0
    ingredients (list): 'RSD', 'damping', 'FOG', 'galactic_bias', 'bias_in_fisher', 'polynomial_in_fisher', 'ap_effect', 'shot_noise', 'reconstruction'
    (deprecated) initial_params (dict of dict): alpha, beta ([value, stdev]). Derived parameters to be constrained.
    cosmological_parameters_in_fisher (list): if not None, will only include these parameters in fisher.
    polynomial_parameters: polynomial corrections of power spectrum to be considered, including a_m, b_n.

    todos:  - add FOG
            - ap effect for BS
            - polynomials for BS
    """
    def __init__(self, cosmological_parameters=None, 
                fiducial_cosmological_parameters=None, fix_H0=False,
                cosmological_parameters_in_fisher=['ombh2', 'omch2', 'As', 'ns', 'tau', 'YHe', 'thetastar', 'nnu']
                ):
        #
        self.camb_cosmology = gc.camb_cosmology(cosmological_parameters, fiducial_cosmological_parameters, fix_H0)
        self.cosmological_parameters = self.camb_cosmology.parameters
        self.fiducial_cosmological_parameters = self.camb_cosmology.fiducial_parameters
        if cosmological_parameters_in_fisher is None:
            self.cosmological_parameters_in_fisher = list(self.cosmological_parameters.keys())
        else:
            self.cosmological_parameters_in_fisher = cosmological_parameters_in_fisher
        self.pisf = pi/self.camb_cosmology.rstar
        self.evaluation_count = 0
        self.camb_cosmology.prepare_power_spectrum_derivative_parts(self.cosmological_parameters_in_fisher)

        #
        r = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5])
        x = np.array([0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0])
        self.r_x = InterpolatedUnivariateSpline(x, r, k=1, ext=3)

        self.alpha = self.camb_cosmology.alpha
        self.beta = self.camb_cosmology.beta
        self.delta_beta = self.beta - df.n2b(self.fiducial_cosmological_parameters['nnu']['value'])


    def update_survey_setups(self, survey_geometrics={'f_sky': 0.5,'N_g': 1e100,'z_min': 0.1,'z_max': 4.0,'dz': 0.1,'Sigma_0': 16.6,'reconstruction_rate': 0.5,'b_0': 0.95,'survey_type':'spectroscopic','sigma8_0': 0.9,}, 
                             ingredients=['RSD', 'damping', 'galactic_bias', 'bias_in_fisher', 'polynomial_in_fisher', 'ap_effect', 'shot_noise', 'reconstruction'], 
                             #initial_params={'alpha': {'value': 1.0,'stdev': 0.008216265109777156}, 'beta': {'value': 1.0,'stdev': 1e10}}, 
                             polynomial_parameters={'a': [0, 1, 2, 3, 4], 'b': [1, 2, 3, 4, 5]}
                             ):
        '''
        Move some parts of __init__ here, to make sure we can update parameters (except cosmological ones) at any time.
        '''
        #
        for key in survey_geometrics:
            setattr(self, key, survey_geometrics[key])
        self.ingredients = ingredients
        # for key in initial_params:
        #     setattr(self, key, initial_params[key])
        self.polynomial_parameters=polynomial_parameters
        if self.polynomial_parameters is None:
            self.polynomial_parameters = {'a': [], 'b': []}
        self.cosmo = self.set_cosmo(sigma8_0=self.sigma8_0)

        #
        if 'ng_z_list' in list(survey_geometrics.keys()):
            self.z_min = self.ng_z_list[0,0] - self.ng_z_list[0,2]/2
            self.z_max = self.ng_z_list[-1,0] + self.ng_z_list[-1,2]/2
            self.z_max_int = self.z_max
        self.V_tot = self.survey_volume(self.f_sky, self.z_min, self.z_max)

        if self.survey_type == 'spectroscopic' or not ('survey_type' in survey_geometrics):
            if 'ng_z_list' in list(survey_geometrics.keys()):
                if len(self.ng_z_list) == 1:
                    ng_temp = self.ng_z_list[0,1]
                    self.ng = lambda x: ng_temp*pow(x, 0)
                else:
                    self.ng = InterpolatedUnivariateSpline(self.ng_z_list[:,0], self.ng_z_list[:,1], k=1)
                self.zmid_list = self.ng_z_list[:,0]
                self.dz_list = self.ng_z_list[:,2]
            else:
                self.ng = lambda x: self.N_g/self.V_tot
                number_z = int(np.floor((self.z_max-self.z_min)/self.dz))
                self.z_max_int = self.z_min+self.dz*number_z
                self.zmid_list = np.linspace(self.z_min+self.dz/2, self.z_max_int-self.dz/2, num=number_z)
                if self.z_max_int != self.z_max:
                    self.zmid_list = np.append(self.zmid_list, (self.z_max+self.z_max_int)/2.0)
                self.dz_list = np.repeat(self.dz, len(self.zmid_list))
        else:
            pass

        #
        ndim_p = 2+len(self.cosmological_parameters_in_fisher)
        ndim_b = 2+len(self.cosmological_parameters_in_fisher)
        if 'bias_in_fisher' in self.ingredients:
            ndim_b += 3
            ndim_p += 1
        if 'polynomial_in_fisher' in self.ingredients:
            ndim_p += len(self.polynomial_parameters['a']) + len(self.polynomial_parameters['b'])
            ndim_b += len(self.polynomial_parameters['a']) + len(self.polynomial_parameters['b'])
        self.db_shape = (ndim_b, ndim_b)
        self.dp_shape = (ndim_p, ndim_p)

        #
        self.set_reconstruction_rate()



    def set_cosmo(self, sigma8_0=0.9):
        h = self.camb_cosmology.h
        Omega_m = (self.cosmological_parameters['ombh2']['value']+self.cosmological_parameters['omch2']['value'])/pow(h, 2)
        Omega_L = 1.-Omega_m
        params = {
            'Omega_m': Omega_m,
            'Omega_L': Omega_L,
            'h': h,
            'sigma8_0': sigma8_0,
        }
        return cosmology(params)



    def survey_volume(self, f_sky, zmin, zmax):
        """
        unit: (Mpc/h)^3
        """
        astropy_cosmo = self.cosmo.astropy_cosmology
        h = self.cosmo.h
        v = (astropy_cosmo.comoving_volume(zmax)-astropy_cosmo.comoving_volume(zmin))*f_sky
        return v.value*h**3


    def reduced_k_and_mu(self, k, mu=0.0, z=0.0, ap_effect=False, phase_shift=False, isotropic=False, fix_rstar=False, external_q_parts=None):
        z = max(z, 1e-3)
        if external_q_parts is not None:
            q_parallel, q_vertical, q_isotropic = external_q_parts
        else:
            q_parallel = self.camb_cosmology.q_parallel
            q_vertical = self.camb_cosmology.q_vertical
            q_isotropic = self.camb_cosmology.q_isotropic
        if ap_effect:
            if isotropic:
                kp = k/q_isotropic(z)
                mup = 0.
            else:
                qp, qv = q_parallel(z), q_vertical(z)
                kp = k*sqrt((1-mu**2)/qv**2+mu**2/qp**2)
                mup = mu/sqrt(mu**2+(1-mu**2)*(qp/qv)**2)
        else:
            kp, mup = k, mu
        if fix_rstar:
            kp *= self.camb_cosmology.alpha
        if phase_shift:
            if not fix_rstar:
                kp /= self.camb_cosmology.alpha
            kp += self.delta_beta*f_phase(kp)/self.camb_cosmology.rstar
        return kp, mup
    
    def galactic_bias(self, double z, bias=False):
        if not bias:
            return 1.0
        bias = self.cosmo.D0/self.cosmo.linear_growth_factor(z)*self.b_0
        return bias

    def galactic_bias_b2(self, double z, bias=False):
        # Ref: https://arxiv.org/pdf/1511.01096.pdf
        if not bias:
            return 0.0
        b1 = self.galactic_bias(z)
        b2 = 0.412 - 2.143* b1 + 0.929* pow(b1, 2) + 0.008* pow(b1, 3)
        #return 0.
        return b2

    def galactic_bias_bs2(self, double z, bias=False):
        # Ref: https://arxiv.org/pdf/1405.1447.pdf (eq. 39)
        if not bias:
            return 0.0
        b1 = self.galactic_bias(z)
        bs2 = 4./7.*(1.-b1)
        #return 0.
        return bs2


    def set_reconstruction_rate(self):
        z_interp_array = np.linspace(1e-5, 10., num=401)
        nz = len(z_interp_array)
        rrs = []
        for z in z_interp_array:
            xx = self.power_spectrum(0.14, mu=0.6, z=0.001, no_wiggle=True)*self.ng(z)/0.1734
            if xx < 0.2:
                rrs.append(1.0)
            elif xx > 10.0:
                rrs.append(0.5)
            else: 
                rrs.append(self.r_x(xx))
        rrs = np.array(rrs)
        self.reconstruction_rate = lambda x: two_value_interpolation_c(z_interp_array, rrs, x, nz)


    def damping_factor(self, k, mu=0.0, z=0.0, damp=False, reconstruction=False):
        if not damp:
            return 1.0

        Sigma_vertical = 9.4* self.cosmo.sigma_8(z)/0.9
        Sigma_parallel = (1+self.cosmo.linear_growth_rate(z))*Sigma_vertical
        if reconstruction:
            rr = self.reconstruction_rate(z)
            Sigma_vertical *= rr
            Sigma_parallel *= rr
        damping = exp(-0.5*pow(k, 2)* (pow(Sigma_vertical, 2)+pow(mu, 2)*(pow(Sigma_parallel, 2)-pow(Sigma_vertical,2))))
        return damping

    def rsd_factor_z1(self, z, mu=0.0, external_biases=None, rsd=False, bias=False):
        """
        Red shift distorsion effects. Use dedicated external bias factors instead if external_biases option on.
        """
        if not rsd:
            fmu2 = 0.
        else:
            fmu2 = self.cosmo.linear_growth_rate(z)*pow(mu, 2)
        if external_biases is None:
            rsd = self.galactic_bias(z, bias=bias) + fmu2
        else:
            rsd = external_biases[0] + fmu2
        return rsd

    def fog_factor(self, k, mu=0.0, fog=False):
        if not fog:
            return 1.0
        fog = exp(-pow(k*mu*self.sigma_p, 2)/2.)
        return fog

    def oscillation_part(self, k, mu=0.0, z=0.0, damp=False, ap_effect=False, phase_shift=False, fix_rstar=False, reconstruction=False, external_ps_parts=None):
        if external_ps_parts is not None:
            _, oscillation_part, _, _, _ = external_ps_parts
        else:
            oscillation_part = self.camb_cosmology.oscillation_part
        k, mu = self.reduced_k_and_mu(k, mu=mu, z=z, ap_effect=ap_effect, phase_shift=phase_shift, fix_rstar=fix_rstar)
        osc = oscillation_part(k)
        osc *= self.damping_factor(k, mu, z, damp=damp, reconstruction=reconstruction)
        return osc
        
    def power_spectrum(self, double k, mu=0.0, z=0.0, debug=False, no_wiggle=False, rsd=False, noise=False, fog=False, 
                        external_biases=None, damp=False, ap_effect=False, reconstruction=False, bias=False, external_ps_parts=None, matter_only=False, ap_effect_wiggle_only=False):
        if not debug:
            rsd = ('RSD' in self.ingredients)
            noise = ('shot_noise' in self.ingredients)
            fog = ('FOG' in self.ingredients)
            damp = ('damping' in self.ingredients)
            ap_effect = ('ap_effect' in self.ingredients)
            reconstruction = ('reconstruction' in self.ingredients)
            bias = ('galactic_bias' in self.ingredients)
        if matter_only:
            rsd = False
            noise = False
            fog = False
            ap_effect = False
            bias = False
        z = max(z, 1e-4)

        if external_ps_parts is not None:
            matter_power_spectrum_no_wiggle, _, q_parallel, q_vertical, q_isotropic = external_ps_parts
        else:
            matter_power_spectrum_no_wiggle = self.camb_cosmology.matter_power_spectrum_no_wiggle
            q_parallel = self.camb_cosmology.q_parallel
            q_vertical = self.camb_cosmology.q_vertical
            q_isotropic = self.camb_cosmology.q_isotropic
        
        # prepare k, mu for broadband
        if ap_effect_wiggle_only:
            # drop external parts, just use the fiducial one.
            k1, mu1 = self.reduced_k_and_mu(k, mu=mu, z=z, ap_effect=False)
        else:
            k1, mu1 = self.reduced_k_and_mu(k, mu=mu, z=z, ap_effect=ap_effect, external_q_parts=(q_parallel, q_vertical, q_isotropic))
        cdef double p = matter_power_spectrum_no_wiggle(k1)

        # parepare k, mu for wiggles
        k, mu = self.reduced_k_and_mu(k, mu=mu, z=z, ap_effect=ap_effect, external_q_parts=(q_parallel, q_vertical, q_isotropic))
        if not no_wiggle:
            # osc. part is replace here inside the function.
            p *= 1 + self.oscillation_part(k, mu, z, damp=damp, reconstruction=reconstruction, external_ps_parts=external_ps_parts)
        if rsd or bias:
            p *= pow(self.rsd_factor_z1(z, mu, external_biases=external_biases, rsd=rsd, bias=bias), 2)
        p *= pow(self.cosmo.linear_growth_factor(z)/self.cosmo.D0, 2)
        if ap_effect:
            p /= q_isotropic(z)**3
        if noise:
            p += 1/self.ng(z)
        return p

    def power_spectrum_derivative_analytical(self, double k, mu=0.0, z=0.0, matter_only=False):
        rsd = ('RSD' in self.ingredients)
        noise = ('shot_noise' in self.ingredients)
        fog = ('FOG' in self.ingredients)
        damp = ('damping' in self.ingredients)
        ap_effect = ('ap_effect' in self.ingredients)
        reconstruction = ('reconstruction' in self.ingredients)
        bias = ('galactic_bias' in self.ingredients)

        if matter_only:
            rsd = False
            noise = False
            fog = False
            ap_effect = False
            bias = False

        k_t, mu_t = self.reduced_k_and_mu(k, mu=mu, z=z, ap_effect=ap_effect, phase_shift=True, isotropic=False, fix_rstar=False)
        dodk = self.camb_cosmology.oscillation_part_derivative(k_t)
        k_t, mu_t = self.reduced_k_and_mu(k, mu=mu, z=z, ap_effect=ap_effect, phase_shift=False, isotropic=False, fix_rstar=False)
        p = self.power_spectrum(k, mu, z, debug=True, no_wiggle=True, rsd=rsd, noise=False, fog=fog, damp=damp, ap_effect=ap_effect, reconstruction=reconstruction, bias=bias)
        dpdk = p*dodk
        dpdk *= self.damping_factor(k_t, mu_t, z, damp=damp, reconstruction=reconstruction)
        dpd_alpha = dpdk*(-k_t/pow(self.camb_cosmology.alpha, 2))
        dpd_beta = dpdk*(f_phase(k_t)/self.camb_cosmology.rstar)
        return np.array([dpd_alpha, dpd_beta])


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def power_spectrum_derivative_bias(self, double k, mu=0., z=0.):
        """
        Calculate the derivative in terms of bias factors numerically.
        """
        if not('bias_in_fisher' in self.ingredients):
            return []
        cdef double b = self.galactic_bias(z)
        biases = [b, -1., 0.]
        cdef double eps = 0.05      # assuming 5% step increment
        dbiases = [eps]
        dPdb = [0.]
        for i in range(len(dPdb)):
            biases_temp = biases
            biases_temp[i] -= dbiases[i]
            Pm = self.power_spectrum(k, mu=mu, z=z, external_biases=tuple(biases_temp))
            biases_temp = biases
            biases_temp[i] += dbiases[i]
            Pp = self.power_spectrum(k, mu=mu, z=z, external_biases=tuple(biases_temp))
            dPdb[i] = (Pp-Pm)/(2.*dbiases[i])
        return np.array(dPdb)


    def power_spectrum_derivative_cosmological_parameters(self, double k, mu=0.0, z=0.0, wiggle_only=False, phase_only=False, matter_only=False):
        '''
        Note phase only here doesn't mean phase shift information, but pure ap effect that only considered in phase of wiggles.
        '''
        dPdp = []
        for key in self.cosmological_parameters_in_fisher:
            dP = 0.
            for pm in ['plus', 'minus']:
                psdp = self.camb_cosmology.power_spectrum_derivative_parts[key][pm]
                if wiggle_only:
                    matter_power_spectrum_no_wiggle = self.camb_cosmology.matter_power_spectrum_no_wiggle
                    oscillation_part = psdp['oscillation_part']
                    # q_parallel = self.camb_cosmology.q_parallel
                    # q_vertical = self.camb_cosmology.q_vertical

                    # or this set
                    q_parallel = psdp['q_parallel']
                    q_vertical = psdp['q_vertical'] 

                    q_isotropic = self.camb_cosmology.q_isotropic
                elif phase_only:
                    matter_power_spectrum_no_wiggle = self.camb_cosmology.matter_power_spectrum_no_wiggle
                    oscillation_part = self.camb_cosmology.oscillation_part
                    # q_parallel = self.camb_cosmology.q_parallel
                    # q_vertical = self.camb_cosmology.q_vertical

                    # or this set
                    q_parallel = psdp['q_parallel']
                    q_vertical = psdp['q_vertical'] 

                    q_isotropic = self.camb_cosmology.q_isotropic
                else:
                    matter_power_spectrum_no_wiggle = psdp['matter_power_spectrum_no_wiggle']
                    oscillation_part = psdp['oscillation_part']
                    q_parallel = psdp['q_parallel']
                    q_vertical = psdp['q_vertical'] 
                    q_isotropic = psdp['q_isotropic']
                Pt = self.power_spectrum(k, mu=mu, z=z, 
                                        external_ps_parts=(matter_power_spectrum_no_wiggle, oscillation_part, q_parallel, q_vertical, q_isotropic), 
                                        matter_only=matter_only,
                                        ap_effect_wiggle_only=np.any([wiggle_only, phase_only]))

                if pm is 'plus':
                    dP += Pt
                else:
                    dP -= Pt
            dPdp.append(dP/(2*self.camb_cosmology.power_spectrum_derivative_parts[key]['h']))
        return np.array(dPdp)

    def power_spectrum_derivative_polynomial(self, double k, mu=0.0, z=0.0, wiggle_only=False, matter_only=False):
        if not('polynomial_in_fisher' in self.ingredients):
            return []
        if not wiggle_only:
            p = self.power_spectrum(k, mu=mu, z=z, matter_only=matter_only)
            o = self.power_spectrum(k, mu=mu, z=z, matter_only=matter_only, external_ps_parts=(lambda x: 1.0, self.camb_cosmology.oscillation_part, 
                                                                    self.camb_cosmology.q_parallel, self.camb_cosmology.q_vertical, self.camb_cosmology.q_isotropic))
            dps = []
            for n in self.polynomial_parameters['a']:
                dps.append(o*pow(k, n))
            for m in self.polynomial_parameters['b']:
                dps.append(p*pow(k, m*2))
            return np.array(dps)
        else:
            p = self.power_spectrum(k, mu=mu, z=z, no_wiggle=True, matter_only=matter_only)
            o = self.power_spectrum(k, mu=mu, z=z, matter_only=matter_only)/p-1.
            dps = []
            for n in self.polynomial_parameters['a']:
                dps.append(p*pow(k, n))
            for m in self.polynomial_parameters['b']:
                dps.append(p*o*pow(k, m*2))
            return np.array(dps)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrand_ps(self, k, mu, z, wiggle_only=False):
        """
        return a matrix
        """
        integrand_dp = np.zeros(self.dp_shape)
        dp_analytical = self.power_spectrum_derivative_analytical(k, mu, z)
        dp_bias = self.power_spectrum_derivative_bias(k, mu=mu, z=z)
        dp_cosmology = self.power_spectrum_derivative_cosmological_parameters(k, mu=mu, z=z, wiggle_only=wiggle_only)
        dp_polynomial = self.power_spectrum_derivative_polynomial(k, mu=mu, z=z, wiggle_only=wiggle_only)
        dp = np.concatenate((dp_analytical, dp_cosmology, dp_bias, dp_polynomial))

        for i in range(int(self.dp_shape[0])):
            for j in range(int(self.dp_shape[1])):
                integrand_dp[i,j] = dp[i]*dp[j]
        integrand_cov = 1/self.power_spectrum(k, mu, z)**2
        integrand = integrand_dp*integrand_cov* k**2
        return integrand

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def naive_integration_ps(self, args, parallel=False):
        res = 0
        if parallel == True:
            pool = pp.ProcessPool(cpu_count())
            z_list = [args for x in self.kmu_list]
            z_list = [x for x, in z_list]
            k_list = [x for x, y in self.kmu_list]
            mu_list = [y for x, y in self.kmu_list]
            results = pool.uimap(self.integrand_ps, k_list, mu_list, z_list)
            #results = [self.integrand_ps(*x) for x in kmuz_list]
            res = sum(list(results))
        else:
            for kmu in self.kmu_list:
                res += self.integrand_ps(*kmu, *args)
        res *= self.dk*self.dmu
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_power_spectrum_fisher_matrix(self, regions=[{'k_min': 0.01,'k_max': 0.5,'mu_min': -1.0,'mu_max': 1.0}],
                                        addprior=False, tol=1e-4, rtol=1e-4, div_k=50, div_mu=20, parallel=False, wiggle_only=False, physical_kmin=True):
        fisher_matrix_ps_list = []
        for iz in range(len(self.zmid_list)):
            z = self.zmid_list[iz]
            dz = self.dz_list[iz]
            fisher_temp = np.zeros(self.dp_shape)
            if z+dz/2 <= self.z_max_int:
                v = self.survey_volume(self.f_sky, z-dz/2, z+dz/2)
            else:
                v = self.survey_volume(self.f_sky, self.z_max_int, self.z_max)

            bin_label = 'zmin%.2f_zmax%.2f'%(z-dz/2, z+dz/2)
            entries = ['alpha', 'beta']
            for k in self.cosmological_parameters_in_fisher:
                entries.append(k)
            if 'bias_in_fisher' in self.ingredients:
                entries.append('bias_b1-%s'%bin_label)
            if 'polynomial_in_fisher' in self.ingredients:
                for n in self.polynomial_parameters['a']:
                    entries.append('ps_poly_a%d-%s'%(n, bin_label))
                for n in self.polynomial_parameters['b']:
                    entries.append('ps_poly_b%d-%s'%(n, bin_label))

            if 'RSD' not in self.ingredients:
                for subregion in regions:
                    k_min = subregion['k_min']
                    if physical_kmin:
                        k_min = max(k_min, 2.*np.pi*pow(3.*v/4./np.pi, -1./3.))
                    k_max = subregion['k_max']
                    if div_k !=0:
                        self.dk = dk = (k_max-k_min)/div_k
                        self.dmu = 1.0
                        k_list = np.linspace(k_min+dk/2, k_max-dk/2, num=div_k)
                        mu_list = np.array([0.0])
                        self.kmu_list = list(itertools.product(k_list, mu_list))
                        fisher_temp += v/(4*pi**2)*self.naive_integration_ps(args=(z, wiggle_only), parallel=parallel)
                    else:
                        fisher_temp += v/(4*pi**2)*integrate.quad(self.integrand_ps, k_min, k_max, args=(0, z,), limit=1000, epsrel=rtol, epsabs=tol)[0]
            else:
                for subregion in regions:
                    k_min = subregion['k_min']
                    if physical_kmin:
                        k_min = max(k_min, 2.*np.pi*pow(3.*v/4./np.pi, -1./3.))
                    k_max = subregion['k_max']
                    mu_min = subregion['mu_min']
                    mu_max = subregion['mu_max']
                    if div_k !=0 and div_mu!=0:
                        self.dk = dk = (k_max-k_min)/div_k
                        self.dmu = dmu = (mu_max-mu_min)/div_mu
                        k_list = np.linspace(k_min+dk/2, k_max-dk/2, num=div_k)
                        mu_list = np.linspace(mu_min+dmu/2, mu_max-dmu/2, num=div_mu)
                        self.kmu_list = list(itertools.product(k_list, mu_list))
                        fisher_temp += v/(8*pi**2)*self.naive_integration_ps(args=(z, wiggle_only), parallel=parallel)
                    else:
                        fisher_temp += v/(8*pi**2)*integrate.dblquad(self.integrand_ps, mu_min, mu_max, lambda mu: k_min, lambda mu: k_max, args=(z,), epsabs=tol, epsrel=rtol)[0]
            fisher_matrix_ps_list.append(fm.fisher(fisher_temp, entries))
        self.fisher_matrix_ps_list = fisher_matrix_ps_list
        nzs = len(fisher_matrix_ps_list)
        fisher_ps = fisher_matrix_ps_list[0]
        for i in range(1, nzs):
            fisher_ps = fisher_ps.merge(fisher_matrix_ps_list[i])
        self.power_spectrum_fisher_matrix = fisher_ps
        return fisher_ps


    def rsd_factor_z2(self, double k1, double k2, double cos12, mu1=0., mu2=0., z=0., external_biases=None, rsd=False, bias=False):
        """
        Z_2 factor for RSD

        todo: check rsd
        """
        cdef double f12, g12, k12, mu12, f, b, b2, bs2, s12, res
        if not rsd:
            mu1 = 0.
            mu2 = 0.
        f12 = f_kernal(k1, k2, cos12)
        g12 = g_kernal(k1, k2, cos12)
        k12 = sqrt(k1*k1+k2*k2+2*k1*k2*cos12)
        mu12 = (k1*mu1+k2*mu2)/k12
        f = self.cosmo.linear_growth_rate(z)
        if external_biases is None:
            b = self.galactic_bias(z, bias=bias)
            b2 = self.galactic_bias_b2(z, bias=bias)
            bs2 = self.galactic_bias_bs2(z, bias=bias)
        else:
            b, b2, bs2 = external_biases
        s12 = cos12*cos12 -1./3.
        res = b2/2 +b*f12 +f*mu12*mu12*g12
        res += f*mu12*k12/2*(mu1/k1*self.rsd_factor_z1(z, mu=mu2, external_biases=external_biases, rsd=rsd, bias=bias) \
                            +mu2/k2*self.rsd_factor_z1(z, mu=mu1, external_biases=external_biases, rsd=rsd, bias=bias))
        res += bs2/2.*s12
        return res




    def bispectrum(self, kargs, muargs=(0., 0., 0.), z=0., coordinate='cartesian', debug=False, no_wiggle=False, rsd=False, noise=False, fog=False,
                    external_biases=None, damp=False, ap_effect=False, reconstruction=False, bias=None, external_ps_parts=None, matter_only=False):
        """
        todos: derive bispectrum in galsurvey, currently the functions should use with no ingredients
        mu1, mu2, mu3 are not all independent
        """
        if not debug:
            rsd = ('RSD' in self.ingredients)
            noise = ('shot_noise' in self.ingredients)
            fog = ('FOG' in self.ingredients) # to do
            damp = ('damping' in self.ingredients)
            ap_effect = ('ap_effect' in self.ingredients) 
            reconstruction = ('reconstruction' in self.ingredients)
            bias = ('galactic_bias' in self.ingredients)
        if matter_only:
            rsd = False
            noise = False
            fog = False
            ap_effect = False
            bias = False
        z = max(z, 1e-4)

        if external_ps_parts is not None:
            _, _, q_parallel, q_vertical, q_isotropic = external_ps_parts
        else:
            q_parallel = self.camb_cosmology.q_parallel
            q_vertical = self.camb_cosmology.q_vertical
            q_isotropic = self.camb_cosmology.q_isotropic
        q3 = pow(q_isotropic(z), 3)

        if coordinate == 'cartesian':
            k_1, k_2, k_3 = kargs
        elif coordinate == 'child18':
            k_1, k_2, k_3 = k_tf(*kargs)
        elif coordinate == 'ascending':
            k_1, k_2, k_3 = k_tf_as(*kargs)
        mu1, mu2, mu3 = muargs
        if ap_effect:
            k_1, mu1 = self.reduced_k_and_mu(k_1, mu1, z=z, ap_effect=True, external_q_parts=(q_parallel, q_vertical, q_isotropic))
            k_2, mu2 = self.reduced_k_and_mu(k_2, mu2, z=z, ap_effect=True, external_q_parts=(q_parallel, q_vertical, q_isotropic))
            k_3, mu3 = self.reduced_k_and_mu(k_3, mu3, z=z, ap_effect=True, external_q_parts=(q_parallel, q_vertical, q_isotropic))
        
        cos12, cos23, cos31 = cost(k_1, k_2, k_3), cost(k_2, k_3, k_1), cost(k_3, k_1, k_2)

        z12 = self.rsd_factor_z2(k_1, k_2, cos12, mu1=mu1, mu2=mu2, z=z, external_biases=external_biases, rsd=rsd, bias=bias)
        z23 = self.rsd_factor_z2(k_2, k_3, cos23, mu1=mu2, mu2=mu3, z=z, external_biases=external_biases, rsd=rsd, bias=bias)
        z31 = self.rsd_factor_z2(k_3, k_1, cos31, mu1=mu3, mu2=mu1, z=z, external_biases=external_biases, rsd=rsd, bias=bias)
        z1 = self.rsd_factor_z1(z, mu=mu1, external_biases=external_biases, rsd=rsd, bias=bias)
        z2 = self.rsd_factor_z1(z, mu=mu2, external_biases=external_biases, rsd=rsd, bias=bias)
        z3 = self.rsd_factor_z1(z, mu=mu3, external_biases=external_biases, rsd=rsd, bias=bias)

        # Note that we only need raw linear matter power spectrum, but wiggles can be damped
        p1 = self.power_spectrum(k_1, mu=mu1, z=z, debug=True, no_wiggle=no_wiggle, rsd=False, noise=False, fog=False, external_biases=None,damp=damp, ap_effect=False,
                                reconstruction=reconstruction, bias=False, external_ps_parts=external_ps_parts)/q3
        p2 = self.power_spectrum(k_2, mu=mu2, z=z, debug=True, no_wiggle=no_wiggle, rsd=False, noise=False, fog=False, external_biases=None,damp=damp, ap_effect=False,
                                reconstruction=reconstruction, bias=False, external_ps_parts=external_ps_parts)/q3
        p3 = self.power_spectrum(k_3, mu=mu3, z=z, debug=True, no_wiggle=no_wiggle, rsd=False, noise=False, fog=False, external_biases=None,damp=damp, ap_effect=False,
                                reconstruction=reconstruction, bias=False, external_ps_parts=external_ps_parts)/q3
        res = 2*(p1*p2*z12*z1*z2 +p2*p3*z23*z2*z3 +p3*p1*z31*z3*z1) *is_zero(beta(cos12)*beta(cos23)*beta(cos31))
        if noise:
            res += (p1+p2+p3)/self.ng(z) + 1/pow(self.ng(z), 2)
        return res

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def bispectrum_derivative_analytical(self, (double, double, double) kargs, muargs=(0., 0., 0.), z=0, coordinate='cartesian'):
        """
        will give different results depending on the coordinate
        """
        rsd = ('RSD' in self.ingredients)
        noise = ('shot_noise' in self.ingredients)
        fog = ('FOG' in self.ingredients)
        damp = ('damping' in self.ingredients)
        ap_effect = ('ap_effect' in self.ingredients)
        reconstruction = ('reconstruction' in self.ingredients)
        bias = ('galactic_bias' in self.ingredients)

        cdef double k_1, k_2, k_3, mu_1, mu_2, mu_3
        if coordinate =='cartesian':
            k_1, k_2, k_3 = kargs
        elif coordinate =='child18':
            k_1, k_2, k_3 = k_tf(*kargs)
        elif coordinate == 'ascending':
            k_1, k_2, k_3 = k_tf_as(*kargs)

        mu1, mu2, mu3 = muargs
        if ap_effect:
            k_1, mu1 = self.reduced_k_and_mu(k_1, mu1, z=z, ap_effect=True)
            k_2, mu2 = self.reduced_k_and_mu(k_2, mu2, z=z, ap_effect=True)
            k_3, mu3 = self.reduced_k_and_mu(k_3, mu3, z=z, ap_effect=True)
            q3 = pow(self.camb_cosmology.q_isotropic(z), 3)
        else:
            q3 = 1.

        cos12, cos23, cos31 = cost(k_1, k_2, k_3), cost(k_2, k_3, k_1), cost(k_3, k_1, k_2)

        if abs(cos12)>1. or abs(cos23)>1. or abs(cos31)>1.:
            return np.array([0.0, 0.0])

        z12 = self.rsd_factor_z2(k_1, k_2, cos12, mu1=mu1, mu2=mu2, z=z, rsd=rsd, bias=bias)
        z23 = self.rsd_factor_z2(k_2, k_3, cos23, mu1=mu2, mu2=mu3, z=z, rsd=rsd, bias=bias)
        z31 = self.rsd_factor_z2(k_3, k_1, cos31, mu1=mu3, mu2=mu1, z=z, rsd=rsd, bias=bias)
        z1 = self.rsd_factor_z1(z, mu=mu1, rsd=rsd, bias=bias)
        z2 = self.rsd_factor_z1(z, mu=mu2, rsd=rsd, bias=bias)
        z3 = self.rsd_factor_z1(z, mu=mu3, rsd=rsd, bias=bias)

        # still, matter only
        p1 = self.power_spectrum(k_1, mu=mu1, z=z, matter_only=True)
        p2 = self.power_spectrum(k_2, mu=mu2, z=z, matter_only=True)
        p3 = self.power_spectrum(k_3, mu=mu3, z=z, matter_only=True)

        dp1 = self.power_spectrum_derivative_analytical(k_1, mu=mu1, z=z, matter_only=True)
        dp2 = self.power_spectrum_derivative_analytical(k_2, mu=mu2, z=z, matter_only=True)
        dp3 = self.power_spectrum_derivative_analytical(k_3, mu=mu3, z=z, matter_only=True)
        
        res = 2*((dp1*p2+p1*dp2)*z12*z1*z2 +(dp2*p3+p2*dp3)*z23*z2*z3 +(dp3*p1+p3*dp1)*z31*z3*z1)/pow(q3, 2)
        return res


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def bispectrum_derivative_bias(self, (double, double, double) kargs, muargs=(0., 0., 0.), z=0, coordinate='cartesian'):
        """
        Calculate the derivative in terms of bias factors numerically.
        """
        if not('bias_in_fisher' in self.ingredients):
            return []
        cdef double b = self.galactic_bias(z)
        cdef double b2 = self.galactic_bias_b2(z)
        cdef double bs2 = self.galactic_bias_bs2(z)
        biases = [b, b2, bs2]
        cdef double eps = 0.05      # assuming 5% step increment
        dbiases = [eps, eps, eps]
        dBdb = np.array([0., 0., 0.])
        for i in range(len(dBdb)):
            biases_temp = biases
            biases_temp[i] -= dbiases[i]
            Bm = self.bispectrum(kargs, muargs=muargs, z=z, coordinate=coordinate, external_biases=tuple(biases_temp))
            biases_temp = biases
            biases_temp[i] += dbiases[i]
            Bp = self.bispectrum(kargs, muargs=muargs, z=z, coordinate=coordinate, external_biases=tuple(biases_temp))
            dBdb[i] = (Bp-Bm)/(2.*dbiases[i])
        return dBdb

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def bispectrum_derivative_cosmological_parameters(self, (double, double, double) kargs, muargs=(0., 0., 0.), z=0, coordinate='cartesian', wiggle_only=False):
        rsd = ('RSD' in self.ingredients)
        noise = ('shot_noise' in self.ingredients)
        fog = ('FOG' in self.ingredients)
        damp = ('damping' in self.ingredients)
        ap_effect = False #('ap_effect' in self.ingredients) # not included so far.
        reconstruction = ('reconstruction' in self.ingredients)
        bias = ('galactic_bias' in self.ingredients)

        if len(self.cosmological_parameters_in_fisher) == 0:
            return []
        cdef double k_1, k_2, k_3, mu_1, mu_2, mu_3

        if coordinate =='cartesian':
            k_1, k_2, k_3 = kargs
        elif coordinate =='child18':
            k_1, k_2, k_3 = k_tf(*kargs)
        elif coordinate == 'ascending':
            k_1, k_2, k_3 = k_tf_as(*kargs)

        mu1, mu2, mu3 = muargs

        if ap_effect:
            k_1, mu1 = self.reduced_k_and_mu(k_1, mu1, z=z, ap_effect=True)
            k_2, mu2 = self.reduced_k_and_mu(k_2, mu2, z=z, ap_effect=True)
            k_3, mu3 = self.reduced_k_and_mu(k_3, mu3, z=z, ap_effect=True)
            q3 = pow(self.camb_cosmology.q_isotropic(z), 3)
        else:
            q3 = 1.

        cos12, cos23, cos31 = cost(k_1, k_2, k_3), cost(k_2, k_3, k_1), cost(k_3, k_1, k_2)

        if abs(cos12)>1 or abs(cos23)>1 or abs(cos31)>1:
            return np.zeros(len(self.cosmological_parameters_in_fisher))
        
        z12 = self.rsd_factor_z2(k_1, k_2, cos12, mu1=mu1, mu2=mu2, z=z, rsd=rsd, bias=bias)
        z23 = self.rsd_factor_z2(k_2, k_3, cos23, mu1=mu2, mu2=mu3, z=z, rsd=rsd, bias=bias)
        z31 = self.rsd_factor_z2(k_3, k_1, cos31, mu1=mu3, mu2=mu1, z=z, rsd=rsd, bias=bias)
        z1 = self.rsd_factor_z1(z, mu=mu1, rsd=rsd, bias=bias)
        z2 = self.rsd_factor_z1(z, mu=mu2, rsd=rsd, bias=bias)
        z3 = self.rsd_factor_z1(z, mu=mu3, rsd=rsd, bias=bias)

        # still, matter only
        p1 = self.power_spectrum(k_1, mu=mu1, z=z, matter_only=True)
        p2 = self.power_spectrum(k_2, mu=mu2, z=z, matter_only=True)
        p3 = self.power_spectrum(k_3, mu=mu3, z=z, matter_only=True)

        dp1 = self.power_spectrum_derivative_cosmological_parameters(k_1, mu=mu1, z=z, matter_only=True, wiggle_only=wiggle_only)
        dp2 = self.power_spectrum_derivative_cosmological_parameters(k_2, mu=mu2, z=z, matter_only=True, wiggle_only=wiggle_only)
        dp3 = self.power_spectrum_derivative_cosmological_parameters(k_3, mu=mu3, z=z, matter_only=True, wiggle_only=wiggle_only)
        
        res = 2*((dp1*p2+p1*dp2)*z12*z1*z2 +(dp2*p3+p2*dp3)*z23*z2*z3 +(dp3*p1+p3*dp1)*z31*z3*z1)/pow(q3, 2)
        return res



    def bispectrum_derivative_polynomial(self, (double, double, double) kargs, muargs=(0., 0., 0.), z=0, coordinate='cartesian', wiggle_only=False):
        """
        This part will be different from the PS one since polynomial definition can be different.
        """
        rsd = ('RSD' in self.ingredients)
        noise = ('shot_noise' in self.ingredients)
        fog = ('FOG' in self.ingredients)
        damp = ('damping' in self.ingredients)
        ap_effect = False #('ap_effect' in self.ingredients) # not included so far.
        reconstruction = ('reconstruction' in self.ingredients)
        bias = ('galactic_bias' in self.ingredients)

        if not('polynomial_in_fisher' in self.ingredients):
            return []

        cdef double k_1, k_2, k_3, mu_1, mu_2, mu_3

        if coordinate =='cartesian':
            k_1, k_2, k_3 = kargs
        elif coordinate =='child18':
            k_1, k_2, k_3 = k_tf(*kargs)
        elif coordinate == 'ascending':
            k_1, k_2, k_3 = k_tf_as(*kargs)

        mu1, mu2, mu3 = muargs

        if ap_effect:
            k_1, mu1 = self.reduced_k_and_mu(k_1, mu1, z=z, ap_effect=True)
            k_2, mu2 = self.reduced_k_and_mu(k_2, mu2, z=z, ap_effect=True)
            k_3, mu3 = self.reduced_k_and_mu(k_3, mu3, z=z, ap_effect=True)
            q3 = pow(self.camb_cosmology.q_isotropic(z), 3)
        else:
            q3 = 1.

        cos12, cos23, cos31 = cost(k_1, k_2, k_3), cost(k_2, k_3, k_1), cost(k_3, k_1, k_2)

        if abs(cos12)>1 or abs(cos23)>1 or abs(cos31)>1:
            return np.zeros(len(self.polynomial_parameters['a'])+len(self.polynomial_parameters['b']))
        
        z12 = self.rsd_factor_z2(k_1, k_2, cos12, mu1=mu1, mu2=mu2, z=z)
        z23 = self.rsd_factor_z2(k_2, k_3, cos23, mu1=mu2, mu2=mu3, z=z)
        z31 = self.rsd_factor_z2(k_3, k_1, cos31, mu1=mu3, mu2=mu1, z=z)
        z1 = self.rsd_factor_z1(z, mu=mu1)
        z2 = self.rsd_factor_z1(z, mu=mu2)
        z3 = self.rsd_factor_z1(z, mu=mu3)

        p1 = self.power_spectrum(k_1, mu=mu1, z=z, matter_only=True)
        p2 = self.power_spectrum(k_2, mu=mu2, z=z, matter_only=True)
        p3 = self.power_spectrum(k_3, mu=mu3, z=z, matter_only=True)

        dp1 = self.power_spectrum_derivative_polynomial(k_1, mu=mu1, z=z, matter_only=True, wiggle_only=wiggle_only)
        dp2 = self.power_spectrum_derivative_polynomial(k_2, mu=mu2, z=z, matter_only=True, wiggle_only=wiggle_only)
        dp3 = self.power_spectrum_derivative_polynomial(k_3, mu=mu3, z=z, matter_only=True, wiggle_only=wiggle_only)
        
        res = 2*((dp1*p2+p1*dp2)*z12*z1*z2 +(dp2*p3+p2*dp3)*z23*z2*z3 +(dp3*p1+p3*dp1)*z31*z3*z1)/pow(q3, 2)

        return res



    def R_bi(self, kargs, muargs=(0.,0.,0.), z=0, coordinate='child18'):
        return self.bispectrum(kargs, muargs=muargs, z=z, coordinate=coordinate)/self.bispectrum(kargs, muargs=muargs, z=z, coordinate=coordinate, nw=True)

    def A_bi(self, delta, theta, k1_min=0.01, k1_max=0.2, div_k1=10000, muargs=(0,0,0), z=0):
        k1sample = np.linspace(k1_min, k1_max, num=div_k1)
        Rsample = self.R_bi(kargs=(k1sample, delta, theta), muargs=muargs, z=z)
        Rmean = np.mean(Rsample)
        #Rmean = integrate.quad(R, kmin_bi, kmax_bi, limit=100)[0]/(kmax_bi-kmin_bi)
        #A2 = lambda x: (R(x)-Rmean)**2
        #A2_val, err = integrate.quad(A2, kmin_bi, kmax_bi, limit=1000)
        A2sample = (Rsample-Rmean)**2
        A2 = np.mean(A2sample)
        return sqrt(A2)

    #@functools.lru_cache(maxsize=None)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrand_bs(self, (double, double, double, double, double) kmuargs, double z, coordinate='cartesian', wiggle_only=False, unique=False, mu_opt=False, double k_max_bi=0.2):
        """
        
        """
        integrand_db = np.zeros(self.db_shape)
        cdef double k1_var, k2_var, k3_var, mu1_var, mu2_var
        cdef double k1, k2, k3, mu1, mu2
        cdef  int i, j

        k1_var, k2_var, k3_var, mu1_var, mu2_var = kmuargs
        cdef (double, double, double) kargs = (k1_var, k2_var, k3_var)

        if coordinate == 'cartesian':
            k1, k2, k3 = kargs
            if beta(cost(*kargs)) == 0.0:
                return np.zeros(self.db_shape)
            cos12 = cost(*kargs)
        elif coordinate == 'child18':
            k1, k2, k3 = k_tf(*kargs)
            cos12 = np.cos(k3_var)
        elif coordinate =='ascending':
            k1, k2, k3 = k_tf_as(*kargs)
            if beta(cost(k1, k2, k3)) == 0.0:
                return np.zeros(self.db_shape)
            cos12 = cost(k1, k2, k3)

        if mu_opt == True:
            mu_s = mu1_var
            mu_r = sqrt(1.-mu_s*mu_s)
            #mu_r = mu1_var
            xi = mu2_var
            mu1, mu2 = mu_tf(mu_r, xi, cos12)
        else:
            mu1, mu2 = mu1_var, mu2_var

        mu3 = -(k1*mu1+k2*mu2)/k3

        if unique==True and (not is_unique(k1, k2, k3)):
            return np.zeros(self.db_shape)
        angular_factor = sigma_angle(mu1, mu2, cos12)
        if 'RSD' in self.ingredients and (angular_factor == 0.) and (not mu_opt):
            return np.zeros(self.db_shape)
        if k1>k_max_bi or k2>k_max_bi or k3>k_max_bi:
            return np.zeros(self.db_shape)

        db_analytical = self.bispectrum_derivative_analytical(kargs, muargs=(mu1, mu2, mu3), z=z, coordinate=coordinate)
        db_bias = self.bispectrum_derivative_bias(kargs, muargs=(mu1, mu2, mu3), z=z, coordinate=coordinate)
        db_cosmology = self.bispectrum_derivative_cosmological_parameters(kargs, muargs=(mu1, mu2, mu3), z=z, coordinate=coordinate, wiggle_only=wiggle_only)
        db_polynomial = self.bispectrum_derivative_polynomial(kargs, muargs=(mu1, mu2, mu3), z=z, coordinate=coordinate, wiggle_only=wiggle_only)
        db = np.concatenate((db_analytical, db_cosmology, db_bias, db_polynomial))

        for i in range(int(self.db_shape[0])):
            for j in range(int(self.db_shape[1])):
                integrand_db[i,j] = db[i]*db[j]
        p1, p2, p3 = self.power_spectrum(k1, mu=mu1, z=z), self.power_spectrum(k2, mu=mu2, z=z), self.power_spectrum(k3, mu=mu3, z=z)
        
        if coordinate in ['cartesian', 'ascending']:
            integrand_cov = k1*k2*k3*beta(cos12)/s123(k1, k2, k3)/(p1*p2*p3)
        elif coordinate == 'child18':    
            integrand_cov = pow(k1*k2, 2)*np.sin(k3_var) *beta(cos12)/s123(k1, k2, k3)/(p1*p2*p3)
        integrand = integrand_db*integrand_cov
        if 'RSD' in self.ingredients:
            if not mu_opt:
                integrand *= angular_factor
            else:
                integrand *= 1/(2*pi)
        self.evaluation_count += 1
        return integrand
            

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def naive_integration_bs(self, args, coordinate='cartesian', method='sobol', unique=False, mu_opt=False, wiggle_only=False):
        cdef int len_kmu = len(self.kkkmu_list)
        ints = np.zeros((len_kmu, *(self.db_shape)))
        #cdef double [:,:,:] res_view = res
        cdef double [:,:,:] ints_view = ints
        cdef int i, j, k

        for i in range(len_kmu):
            arr = self.integrand_bs(self.kkkmu_list[i], *args, coordinate=coordinate, wiggle_only=wiggle_only, unique=unique, mu_opt=mu_opt, k_max_bi=self.k_max_bi)
            for j in range(self.db_shape[0]):
                for k in range(self.db_shape[1]):
                    ints_view[i,j,k] = arr[j,k]

        if method in ['naive', 'monte_carlo', 'sobol']:
            return np.sum(ints, axis=0)*np.prod(self.dds)
        
        if method in ['simpson', 'trapezoidal']:
            if method == 'simpson':
                int_func = integrate.simps
            if method == 'trapezoidal':
                int_func = integrate.trapz
            
            ints = ints.reshape(*(self.divs), *(self.db_shape))
            ints = int_func(ints, self.k1_list, axis=0)
            ints = int_func(ints, self.k2_list, axis=0)
            ints = int_func(ints, self.k3_list, axis=0)


            if 'RSD' in self.ingredients:
                ints = int_func(ints, self.mu1_list, axis=0)
                ints = int_func(ints, self.mu2_list, axis=0)
            else:
                ints = ints[0, 0]
            return ints

    def integrand_2d(self, args, k1_min=0.01, k1_max=0.2, div_k1=19, z=0, coordinate='child18', unique=False, mu_opt=False, integrate_over_mu=False):
        """
        args: k2_var, k3_var, mu1, mu2
        """
        res = 0.0
        dk1 = (k1_max-k1_min)/div_k1
        k1_list = np.linspace(k1_min+dk1/2, k1_max-dk1/2, num=div_k1)
        if not integrate_over_mu:
            for k1 in k1_list:
                res += self.integrand_bs((k1, *args), z=z, coordinate=coordinate, unique=unique, mu_opt=mu_opt)
            return res*dk1
        else:
            div_mu1 = 10
            div_mu2 = 10
            dmu1 = 1/div_k1
            dmu2 = 2*pi/div_k1
            mu1_list = np.linspace(dmu1/2, 1-dmu1/2, num=div_mu1)
            mu2_list = np.linspace(dmu2/2, 2*pi-dmu2/2, num=div_mu2)
            k2_list = [args[0]]
            k3_list = [args[1]]
            kkkmu_list = list(itertools.product(k1_list, k2_list, k3_list, mu1_list, mu2_list))
            for kkkmu in kkkmu_list:
                res += self.integrand_bs(kkkmu, z=z, coordinate=coordinate, unique=unique, mu_opt=True)
            return res*dk1*dmu1*dmu2
            

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_bispectrum_fisher_matrix(self, regions = [{'coordinate': 'cartesian',\
                                            'mu_opt': True,\
                                            'bounds': ((0.01, 0.2),(0.01, 0.2),(0.01, 0.2),(0, 1),(0, 6.283185307179586)),\
                                            'divideby': 'num',\
                                            'divs': (10, 10, 10, 10, 10)}], 
                                    method='sobol', addprior=False, tol=1e-4, rtol=1e-4, unique=True, wiggle_only=False, verbose=False, k_max_bi=2333.):
        """
        integration methods: naive, monte_carlo, simpson, trapezoidal, sobol
        """
        fisher_matrix_bs_list = []

        self.evaluation_count = 0
        self.sampling_count = 0
        self.k_max_bi = k_max_bi
        
        for iz in range(len(self.zmid_list)):
            z = self.zmid_list[iz]
            dz = self.dz_list[iz]

            bin_label = 'zmin%.2f_zmax%.2f'%(z-dz/2, z+dz/2)
            entries = ['alpha', 'beta']
            for k in self.cosmological_parameters_in_fisher:
                entries.append(k)
            if 'bias_in_fisher' in self.ingredients:
                entries.append('bs_bias_b1-%s'%bin_label)
                entries.append('bs_bias_b2-%s'%bin_label)
                entries.append('bs_bias_bs2-%s'%bin_label)

            fisher_temp = np.zeros(self.db_shape)
            if z+dz/2 <= self.z_max_int:
                v = self.survey_volume(self.f_sky, z-dz/2, z+dz/2)
            else:
                v = self.survey_volume(self.f_sky, self.z_max_int, self.z_max)

            for subregion in regions:
                bounds = np.array(subregion['bounds'])
                if subregion['divideby'] == 'num':
                    self.divs = np.array(subregion['divs'])
                    self.dds = np.diff(bounds).flatten()/np.array(self.divs)
                    
                elif subregion['divideby'] == 'step':
                    self.dds = np.array(subregion['dds'])
                    self.divs = np.round(np.diff(bounds).flatten()/np.array(self.dds))
                    self.divs = (self.divs).astype(int)
                
                if method in ['simpson', 'trapezoidal']:
                    self.divs += 1
                
                if 'RSD' not in self.ingredients:
                    bounds[-1] = [0., 0.]
                    bounds[-2] = [0., 0.]
                    self.divs[-1] = 1
                    self.divs[-2] = 1
                    self.dds[-1] = 1.0
                    self.dds[-2] = 1.0
                

                keys = ['k1', 'k2', 'k3', 'mu1', 'mu2']

                if method in ['naive', 'simpson', 'trapezoidal', 'monte_carlo']:
                    i = -1
                    for key in keys:
                        i += 1
                        if method in ['naive']:
                            temp_list = np.linspace((bounds[:,0]+self.dds/2)[i], (bounds[:,1]-self.dds/2)[i], num=self.divs[i])
                            if 'RSD' not in self.ingredients and i>2:
                                temp_list = np.array([0.0])
                        if method in ['simpson', 'trapezoidal']:
                            temp_list = np.linspace(bounds[:,0][i], bounds[:,1][i], num=self.divs[i])
                        if method in ['monte_carlo']:
                            temp_list = stats.uniform.rvs(loc=bounds[:,0][i], scale=bounds[:,1][i]-bounds[:,0][i], size=self.divs[i])    
                        
                        if i<3 and self.divs[i]==1:
                            temp_list = np.array([bounds[:,0][i]/2 + bounds[:,1][i]/2])
                        
                        setattr(self, key+'_list', temp_list)

                    kkkmu_list = list(itertools.product(self.k1_list, self.k2_list, self.k3_list, self.mu1_list, self.mu2_list))
                    
                    
                if method == 'sobol':
                    nd_list = sobol_seq.i4_sobol_generate(len(keys), np.prod(self.divs))
                    for i in range(len(keys)):
                        temp_list = nd_list[:,i]*(bounds[i,1]-bounds[i,0])+bounds[i,0]

                        if i<3 and self.divs[i] == 1:
                            temp_list = np.repeat(bounds[:,0][i]/2 + bounds[:,1][i]/2, len(nd_list))

                        setattr(self, keys[i]+'_list', temp_list)
                    kkkmu_list = list(zip(self.k1_list, self.k2_list, self.k3_list, self.mu1_list, self.mu2_list))

                self.kkkmu_list = kkkmu_list
                fisher_temp += v/(8*pi**4)*self.naive_integration_bs(args=(z,), coordinate=subregion['coordinate'], method=method, wiggle_only=wiggle_only, unique=unique, mu_opt=subregion['mu_opt'])
                self.sampling_count += np.prod(self.divs)

            fisher_matrix_bs_list.append(fm.fisher(fisher_temp, entries))
            if verbose is True:
                print('%.3f'%z, fisher_temp.flatten())
        
        self.fisher_matrix_bs_list = fisher_matrix_bs_list
        nzs = len(fisher_matrix_bs_list)
        fisher_bs = fisher_matrix_bs_list[0]
        for i in range(1, nzs):
            fisher_bs = fisher_bs.merge(fisher_matrix_bs_list[i])
        self.bispectrum_fisher_matrix = fisher_bs
        return fisher_bs


