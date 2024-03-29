# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:03:10 2020

Cosmological Galaxy Survey

@author: yanlong@caltech.edu
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
#from astropy.cosmology import Planck15
from scipy import interpolate
from scipy import misc
from scipy import integrate
from scipy import stats
import functools
import itertools
import pathos.pools as pp
from multiprocessing import cpu_count, Pool
import time


def f_phase(k): 
    return 0.227*k**0.872/(k**0.872+0.0324**0.872)

def k_tf(k1, delta, theta):
    return k1, k1+delta, np.sqrt(k1**2+(k1+delta)**2+2*k1*(k1+delta)*np.cos(theta))

def k_tf_as(k1, delta1, delta2):
    return k1, k1+delta1, k1+delta1+delta2


def mu_tf(mu_r, xi, cos12):
    mu1_t = mu_r*np.cos(xi)
    mu2_t = mu_r*np.sin(xi)
    mu1_t *= np.sqrt(2*(1-cos12))
    mu2_t *= np.sqrt(2*(1+cos12))
    mu1 = (mu1_t+mu2_t)/2
    mu2 = (-mu1_t+mu2_t)/2
    return mu1, mu2


def beta(x):
    if x==1 or x==-1:
        return .5
    elif -1<x<1:
        return 1
    else:
        return 0
beta = np.vectorize(beta)


def is_zero(x):
    if x ==0:
        return 0.0
    else:
        return 1.0
is_zero = np.vectorize(is_zero)

def is_unique(k1, k2, k3):
    """
    This is actually a region function
    """
    return k1<=k2 and k2<=k3
is_unique = np.vectorize(is_unique)

def cost(k1, k2, k3):
    return (k3**2-k1**2-k2**2)/(2*k1*k2)

def f_kernal(k1, k2, cos12):
    return 5./7.+.5*(k1/k2+k2/k1)*cos12+2./7.*cos12**2

def g_kernal(k1, k2, cos12):
    return 3./7.+.5*(k1/k2+k2/k1)*cos12+4./7.*cos12**2

def sigma_angle(mu1, mu2, cos12):
    res = 1 - cos12**2 -mu1**2 - mu2**2 + 2*mu1*mu2*cos12
    if res <= 0.:
        return 0.0
    res = 1/(2*np.pi* np.sqrt(res))
    return res

sigma_angle = np.vectorize(sigma_angle)

def s123(k1, k2, k3):
    if k1==k2==k3:
        return 6
    elif k1==k2 or k2==k3 or k3==k1:
        return 2
    else:
        return 1
    

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

    def linear_growth_rate(self, z):
        Om = self.Omega_m
        Ol = self.Omega_L
        f_growth = (Om*(1+z)**3/(Om*(1+z)-(Om+Ol-1)*(1+z)**2+Ol))**(4./7.)
        return f_growth

    def linear_growth_factor(self, z):
        Om = self.Omega_m
        Ol = self.Omega_L
        E = np.sqrt(Om*(1+z)**3 +(1-Om-Ol)*(1+z)**2 +Ol)
        Omega_z = Om*(1+z)**3/E**2
        lambda_z = Ol/E**2
        D_growth = 2.5/(1+z)*Omega_z/(Omega_z**(4./7.)-lambda_z+(1+Omega_z/2)*(1+lambda_z/70))
        return D_growth

    def comoving_distance(self, z):
        """
        unit: Mpc/h
        """
        return(self.astropy_cosmology.comoving_distance(z).value*self.h)

    def sigma_8(self, z):
        return self.sigma8_0*self.linear_growth_factor(z)/self.D0


class ps_interpolation:
    """
    Intepolates the power spectrum from data.
    ps, ps_nw: txt files
    """
    def __init__(self, ps_file, ps_nw_file=None, kind='linear'):
        pdata = np.loadtxt(ps_file)
        pnwdata = np.loadtxt(ps_nw_file)
        pdata_log = np.log10(pdata)
        p_func_loglog = interpolate.interp1d(pdata_log[:,0], pdata_log[:,1], fill_value="extrapolate", kind=kind)
        self.matter_power_spectrum = lambda x: 10**(p_func_loglog(np.log10(x)))
        pnwdata_log = np.log10(pnwdata)
        pnw_func_loglog = interpolate.interp1d(pnwdata_log[:,0], pnwdata_log[:,1], fill_value="extrapolate", kind=kind)
        self.matter_power_spectrum_no_wiggle = lambda x: 10**(pnw_func_loglog(np.log10(x)))
        oscdata = np.transpose([pdata[:,0], pdata[:,1]/pnwdata[:,1]-1])
        osc = interpolate.interp1d(oscdata[:,0], oscdata[:,1], fill_value="extrapolate", kind=kind)
        self.oscillation_part = osc
        #self.matter_power_spectrum = lambda x: self.matter_power_spectrum_no_wiggle(x)*(1+osc(x))


class survey:
    """
    cosmo: cosmology class
    ps: ps_interpolation class
    survey geometrics (dict): f_sky, N_g, z_min, z_max, dz, ng_z_list ([zmid_list, ng_list])
    survey parameters (dict): Sigma_0, reconstruction_rate, b_0, survey_type
    ingredients (list): 'RSD', 'damping', 'FOG', 'galactic_bias'
    priors (dict of dict): alpha_prior, beta_prior ([mean, stdev])

    todos:  - add a fiducial cosmology
            - add survey_type
            - try other integration methods: e.g., simps
            - add FOG
    """
    def __init__(self, cosmo, ps, survey_geometrics, survey_parameters, ingredients, priors):
        self.cosmo = cosmo
        self.pisf = np.pi/self.cosmo.s_f
        self.ps = ps
        for key in survey_geometrics:
            setattr(self, key, survey_geometrics[key])
        for key in survey_parameters:
            setattr(self, key, survey_parameters[key])
        self.ingredients = ingredients
        for key in priors:
            setattr(self, key, priors[key])

        self.evaluation_count = 0
        self.k_max_bi = 2333.

    #def get_ready(self):
        if hasattr(self, 'ng_z_list'):
            self.z_min = self.ng_z_list[0,0] - self.dz/2
            self.z_max = self.ng_z_list[-1,0] + self.dz/2
            self.z_max_int = self.z_max
        self.V_tot = self.survey_volume(self.f_sky, self.z_min, self.z_max)

        if self.survey_type == 'spectroscopic':
            if hasattr(self, 'ng_z_list'):
                self.ng = interpolate.interp1d(self.ng_z_list[:,0], self.ng_z_list[:,1], fill_value="extrapolate")
                self.zmid_list = self.ng_z_list[:,0]
            else:
                self.ng = lambda x: self.N_g/self.V_tot
                number_z = int(np.floor((self.z_max-self.z_min)/self.dz))
                self.z_max_int = self.z_min+self.dz*number_z
                self.zmid_list = np.linspace(self.z_min+self.dz/2, self.z_max_int-self.dz/2, num=number_z)
                if self.z_max_int != self.z_max:
                    self.zmid_list = np.append(self.zmid_list, (self.z_max+self.z_max_int)/2.0)
        else:
            pass

        r = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5])
        x = np.array([0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0])
        self.r_x = interpolate.interp1d(x, r, fill_value="extrapolate")
        
    def survey_volume(self, f_sky, zmin, zmax):
        """
        unit: (Mpc/h)^3
        """
        astropy_cosmo = self.cosmo.astropy_cosmology
        h = self.cosmo.h
        v = (astropy_cosmo.comoving_volume(zmax)-astropy_cosmo.comoving_volume(zmin))*f_sky
        return v.value*h**3
    
    def galactic_bias(self, z):
        if 'galactic_bias' not in self.ingredients:
            return 1.0
        bias = self.cosmo.D0/self.cosmo.linear_growth_factor(z)*self.b_0
        return bias

    def damping_factor(self, k, mu=0.0, z=0.0):
        if 'damping' not in self.ingredients:
            return 1.0

        Sigma_vertical = 9.4* self.cosmo.sigma_8(z)/0.9
        Sigma_parallel = (1+self.cosmo.linear_growth_rate(z))*Sigma_vertical
        if 'reconstruction' in self.ingredients:
            xx = self.power_spectrum(0.14, mu=0.6, z=0, nw=True)*self.ng(z)/0.1734
            if xx < 0.2:
                self.reconstruction_rate = 1.0
            elif xx > 10.0:
                self.reconstruction_rate = 0.5
            else: 
                self.reconstruction_rate = self.r_x(xx)
            Sigma_vertical *= self.reconstruction_rate
            Sigma_parallel *= self.reconstruction_rate
        #Sigma_vertical = self.Sigma_0*self.cosmo.linear_growth_factor(z)*self.reconstruction_rate
        #Sigma_parallel = self.Sigma_0*self.cosmo.linear_growth_factor(z)*(1+self.cosmo.linear_growth_rate(z))*self.reconstruction_rate
        damping = np.exp(-0.5*k**2* (Sigma_vertical**2+mu**2*(Sigma_parallel**2-Sigma_vertical**2)))
        return damping

    def rsd_factor_z1(self, z, mu=0.0):
        if 'RSD' not in self.ingredients:
            return 1.0
        rsd = self.galactic_bias(z) + self.cosmo.linear_growth_rate(z)*mu**2
        return rsd

    def fog_factor(self, k, mu=0.0):
        if 'FOG' not in self.ingredients:
            return 1.0
        fog = np.exp(-(k*mu*self.sigma_p)**2/2.)
        return fog

    def oscillation_part(self, k, mu=0.0, z=0.0, damp=True, priors=True):
        if priors == True:
            k = k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f
        osc = self.ps.oscillation_part(k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f)
        osc *= (self.damping_factor(k, mu, z) if damp==True else 1.0)
        return osc
        
    def power_spectrum(self, k, mu=0.0, z=0.0, nw=False, linear=False, noise=False):
        p = self.ps.matter_power_spectrum_no_wiggle(k)
        if nw==False:
            p *= 1 + self.oscillation_part(k, mu, z)
        if linear==False:
            p *= self.rsd_factor_z1(z, mu)**2
        p *= (self.cosmo.linear_growth_factor(z)/self.cosmo.D0)**2
        if noise==True:
            p += 1/self.ng(z)
        return p

    def power_spectrum_derivative(self, k, mu=0.0, z=0.0, linear=False):
        k_t = k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f
        dodk = misc.derivative(self.ps.oscillation_part, k_t, dx=1e-6)
        p = self.ps.matter_power_spectrum_no_wiggle(k)
        dpdk = p*dodk
        if linear==False:
            dpdk *= self.rsd_factor_z1(z, mu)**2
        dpdk *= (self.cosmo.linear_growth_factor(z)/self.cosmo.D0)**2
        dpdk *= self.damping_factor(k, mu, z)
        dpd_alpha = dpdk*(-k/self.alpha_prior['mean']**2)
        dpd_beta = dpdk*(f_phase(k)/self.cosmo.s_f)
        return np.array([dpd_alpha, dpd_beta])

    @functools.lru_cache(maxsize=None)
    def integrand_ps(self, k, mu, z, simplify=False, noise=True):
        """
        return a matrix
        """
        if simplify is True:
            integrand_do = np.zeros((2,2))
            k_t = k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f
            dodk = misc.derivative(self.ps.oscillation_part, k_t, dx=1e-6)
            damp = self.damping_factor(k, mu, z)
            dodk *= damp
            dod_alpha = dodk*(-k/self.alpha_prior['mean']**2)
            dod_beta = dodk*(f_phase(k)/self.cosmo.s_f)
            do = np.array([dod_alpha, dod_beta])
            #integrand_do = do[i]*do[j]
            for i in range(2):
                for j in range(2):
                    integrand_do[i,j] = do[i]*do[j]
            integrand_cov = 1/(1+self.ps.oscillation_part(k)*damp)**2/(1+1/(self.ng(z)*self.power_spectrum(k, mu, z)))**2
            integrand = integrand_do*integrand_cov* k**2
            return integrand
        integrand_dp = np.zeros((2,2))
        dp = self.power_spectrum_derivative(k, mu, z)
        #integrand_dp = dp[i]*dp[j]
        for i in range(2):
            for j in range(2):
                integrand_dp[i,j] = dp[i]*dp[j]
        integrand_cov = 1/self.power_spectrum(k, mu, z, noise=noise)**2
        integrand = integrand_dp*integrand_cov* k**2
        return integrand


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

    def fisher_matrix_ps(self, regions, addprior=True, tol=1e-4, rtol=1e-4, div_k=0, div_mu=0, parallel=False):
        """
        input: 1d numpy arrays
        refer __init__():
            a) ng_z_list defined: ng = ng(z)
            b) -- not defined: treat ng = Ng/V
        """
        fisher_ps_list = np.zeros((len(self.zmid_list), 2, 2))
        
        for z in self.zmid_list:
            fisher_temp = np.zeros((2,2))
            if z+self.dz/2 <= self.z_max_int:
                v = self.survey_volume(self.f_sky, z-self.dz/2, z+self.dz/2)
            else:
                v = self.survey_volume(self.f_sky, self.z_max_int, self.z_max)
            
            if 'RSD' not in self.ingredients:
                for subregion in regions:
                    k_min = subregion['k_min']
                    k_max = subregion['k_max']
                    if div_k !=0:
                        self.dk = dk = (k_max-k_min)/div_k
                        self.dmu = 1.0
                        k_list = np.linspace(k_min+dk/2, k_max-dk/2, num=div_k)
                        mu_list = np.array([0.0])
                        self.kmu_list = list(itertools.product(k_list, mu_list))
                        fisher_temp += v/(4*np.pi**2)*self.naive_integration_ps(args=(z,), parallel=parallel)
                    else:
                        fisher_temp += v/(4*np.pi**2)*integrate.quad(self.integrand_ps, k_min, k_max, args=(0, z,), limit=1000, epsrel=rtol, epsabs=tol)[0]
            else:
                for subregion in regions:
                    k_min = subregion['k_min']
                    k_max = subregion['k_max']
                    mu_min = subregion['mu_min']
                    mu_max = subregion['mu_max']
                    if div_k !=0 and div_mu!=0:
                        self.dk = dk = (k_max-k_min)/div_k
                        self.dmu = dmu = (mu_max-mu_min)/div_mu
                        k_list = np.linspace(k_min+dk/2, k_max-dk/2, num=div_k)
                        mu_list = np.linspace(mu_min+dmu/2, mu_max-dmu/2, num=div_mu)
                        self.kmu_list = list(itertools.product(k_list, mu_list))
                        fisher_temp += v/(8*np.pi**2)*self.naive_integration_ps(args=(z,), parallel=parallel)
                    else:
                        fisher_temp += v/(8*np.pi**2)*integrate.dblquad(self.integrand_ps, mu_min, mu_max, lambda mu: k_min, lambda mu: k_max, args=(z,), epsabs=tol, epsrel=rtol)[0]
            #fisher_temp[1, 0] = fisher_temp[0, 1]
            #print(fisher_temp)
            fisher_ps_list[self.zmid_list==z] = fisher_temp
            #print('\t', z, v, fisher_temp[1,1], self.naive_integration_ps(args=(z,), parallel=parallel))
        self.fisher_ps_list = np.array(fisher_ps_list)
        self.fisher_ps = np.sum(fisher_ps_list, axis=0)
        if addprior == True:
            self.fisher_ps[0,0] += 1/self.alpha_prior['stdev']**2
            self.fisher_ps[1,1] += 1/self.beta_prior['stdev']**2
        fisher_ps_inv = np.linalg.inv(self.fisher_ps)
        self.alpha_stdev = np.sqrt(fisher_ps_inv[0,0])
        self.beta_stdev = np.sqrt(fisher_ps_inv[1,1])
        return self.fisher_ps


    def rsd_factor_z2(self, k1, k2, cos12, mu1=0, mu2=0, z=0):
        """
        Z_2 factor for RSD
        """
        f12 = f_kernal(k1, k2, cos12)
        g12 = g_kernal(k1, k2, cos12)
        k12 = np.sqrt(k1**2+k2**2+2*k1*k2*cos12)
        mu12 = (k1*mu1+k2*mu2)/k12
        f = self.cosmo.linear_growth_rate(z)
        b = self.galactic_bias(z)
        b2 = 0.0
        bs2 = 0.0
        s12 = cos12**2 -1./3.
        res = b2/2 +b*f12 +f*mu12**2*g12
        res += f*mu12*k12/2*(mu1/k1*self.rsd_factor_z1(z, mu=mu2) +mu2/k2*self.rsd_factor_z1(z, mu=mu1))
        res += bs2/2.*s12
        return res




    def bispectrum(self, kargs, muargs=(0., 0., 0.), z=0., coordinate='cartesian', nw=False, noise=False):
        """
        todos: derive bispectrum in galsurvey, currently the functions should use with no ingredients
        mu1, mu2, mu3 are not all independent
        """
        if coordinate == 'cartesian':
            k_1, k_2, k_3 = kargs
        elif coordinate == 'child18':
            k_1, k_2, k_3 = k_tf(*kargs)
        elif coordinate == 'ascending':
            k_1, k_2, k_3 = k_tf_as(*kargs)
        mu1, mu2, mu3 = muargs
        cos12, cos23, cos31 = cost(k_1, k_2, k_3), cost(k_2, k_3, k_1), cost(k_3, k_1, k_2)

        z12 = self.rsd_factor_z2(k_1, k_2, cos12, mu1=mu1, mu2=mu2, z=z)
        z23 = self.rsd_factor_z2(k_2, k_3, cos23, mu1=mu2, mu2=mu3, z=z)
        z31 = self.rsd_factor_z2(k_3, k_1, cos31, mu1=mu3, mu2=mu1, z=z)
        z1 = self.rsd_factor_z1(z, mu=mu1)
        z2 = self.rsd_factor_z1(z, mu=mu2)
        z3 = self.rsd_factor_z1(z, mu=mu3)

        p1 = self.power_spectrum(k_1, mu=mu1, z=z, nw=nw, linear=True)
        p2 = self.power_spectrum(k_2, mu=mu2, z=z, nw=nw, linear=True)
        p3 = self.power_spectrum(k_3, mu=mu3, z=z, nw=nw, linear=True)
        res = 2*(p1*p2*z12*z1*z2 +p2*p3*z23*z2*z3 +p3*p1*z31*z3*z1) *is_zero(beta(cos12)*beta(cos23)*beta(cos31))
        if noise==True:
            res += (p1+p2+p3)/self.ng(z) + 1/self.ng(z)**2
        return res

    def bispectrum_derivative(self, kargs, muargs=(0., 0., 0.), z=0, coordinate='cartesian', nw=False, noise=False):
        """
        will give different results depending on the coordinate
        """
        if coordinate =='cartesian':
            k_1, k_2, k_3 = kargs
        elif coordinate =='child18':
            k_1, k_2, k_3 = k_tf(*kargs)
        elif coordinate == 'ascending':
            k_1, k_2, k_3 = k_tf_as(*kargs)

        mu1, mu2, mu3 = muargs
        cos12, cos23, cos31 = cost(k_1, k_2, k_3), cost(k_2, k_3, k_1), cost(k_3, k_1, k_2)

        if np.abs(cos12)>1 or np.abs(cos23)>1 or np.abs(cos31)>1:
            return np.array([0.0, 0.0])
        
        z12 = self.rsd_factor_z2(k_1, k_2, cos12, mu1=mu1, mu2=mu2, z=z)
        z23 = self.rsd_factor_z2(k_2, k_3, cos23, mu1=mu2, mu2=mu3, z=z)
        z31 = self.rsd_factor_z2(k_3, k_1, cos31, mu1=mu3, mu2=mu1, z=z)
        z1 = self.rsd_factor_z1(z, mu=mu1)
        z2 = self.rsd_factor_z1(z, mu=mu2)
        z3 = self.rsd_factor_z1(z, mu=mu3)

        #print(z1, z2, z3, z12, z23, z31)

        p1 = self.power_spectrum(k_1, mu=mu1, z=z, nw=nw, linear=True)
        p2 = self.power_spectrum(k_2, mu=mu2, z=z, nw=nw, linear=True)
        p3 = self.power_spectrum(k_3, mu=mu3, z=z, nw=nw, linear=True)

        dp1 = self.power_spectrum_derivative(k_1, mu=mu1, z=z, linear=True)
        dp2 = self.power_spectrum_derivative(k_2, mu=mu2, z=z, linear=True)
        dp3 = self.power_spectrum_derivative(k_3, mu=mu3, z=z, linear=True)
        
        res = 2*((dp1*p2+p1*dp2)*z12*z1*z2 +(dp2*p3+p2*dp3)*z23*z2*z3 +(dp3*p1+p3*dp1)*z31*z3*z1)
        if noise==True:
            res += (dp1+dp2+dp3)/self.ng(z)
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
        return np.sqrt(A2)

    #@functools.lru_cache(maxsize=None)
    def integrand_bs(self, kmuargs, z, coordinate='cartesian', simplify=False, noise=True, unique=False, mu_opt=False):
        """
        """
        integrand_db = np.zeros((2, 2))

        k1_var, k2_var, k3_var, mu1_var, mu2_var = kmuargs
        kargs = (k1_var, k2_var, k3_var)

        if coordinate == 'cartesian':
            k1, k2, k3 = kargs
            if beta(cost(*kargs)) == 0.0:
                return np.zeros((2,2))
            cos12 = cost(*kargs)
        elif coordinate == 'child18':
            k1, k2, k3 = k_tf(*kargs)
            cos12 = np.cos(k3_var)
        elif coordinate =='ascending':
            k1, k2, k3 = k_tf_as(*kargs)
            if beta(cost(k1, k2, k3)) == 0.0:
                return np.zeros((2,2))
            cos12 = cost(k1, k2, k3)

        if mu_opt == True:
            mu_s = mu1_var
            mu_r = np.sqrt(1.-mu_s**2)
            #mu_r = mu1_var
            xi = mu2_var
            mu1, mu2 = mu_tf(mu_r, xi, cos12)
        else:
            mu1, mu2 = mu1_var, mu2_var

        mu3 = - (k1*mu1+k2*mu2)/k3

        if unique==True and (not is_unique(k1, k2, k3)):
            return np.zeros((2, 2))
        angular_factor = sigma_angle(mu1, mu2, cos12)
        if 'RSD' in self.ingredients and (angular_factor == 0.) and (not mu_opt):
            #print(1 - cos12**2 -mu1**2 - mu2**2 + 2*mu1*mu2*cos12)
            return np.zeros((2, 2))
        if k1>self.k_max_bi or k2>self.k_max_bi or k3>self.k_max_bi:
            return np.zeros((2, 2))

        db = self.bispectrum_derivative(kargs, muargs=(mu1, mu2, mu3), z=z, coordinate=coordinate, noise=noise)
        for i in range(2):
            for j in range(2):
                integrand_db[i,j] = db[i]*db[j]
        p1, p2, p3 = self.power_spectrum(k1, mu=mu1, z=z, noise=noise), self.power_spectrum(k2, mu=mu2, z=z, noise=noise), self.power_spectrum(k3, mu=mu3, z=z, noise=noise)
        
        if coordinate in ['cartesian', 'ascending']:
            integrand_cov = k1*k2*k3*beta(cos12)/s123(k1, k2, k3)/(p1*p2*p3)
        elif coordinate == 'child18':    
            integrand_cov = (k1*k2)**2*np.sin(k3_var) *beta(cos12)/s123(k1, k2, k3)/(p1*p2*p3)
        integrand = integrand_db*integrand_cov
        if 'RSD' in self.ingredients:
            if mu_opt == False:
                integrand *= angular_factor
                #integrand = sigma_angle(mu1, mu2, cos12)
            else:
                #integrand *= sigma_angle(mu1, mu2, cos12) * (1-cos12**2)**1. * mu_r
                integrand *= 1/(2*np.pi)
        self.evaluation_count += 1
        return integrand
            

    #@functools.lru_cache(maxsize=None)
    def naive_integration_bs(self, args, coordinate='cartesian', method='naive', unique=False, mu_opt=False):
        """
        todos:
            - use differrent methods of integration:
                -trapz
                -simps
                -monte carlo
        """
        #print(method)
        if method in ['naive', 'monte_carlo']:
            res = 0
            for kmuargs in self.kkkmu_list:
                res += self.integrand_bs(kmuargs, *args, coordinate=coordinate, unique=unique, mu_opt=mu_opt)
            #print(self.dds)
            return res*np.prod(self.dds)
        
        if method in ['simpson', 'trapezoidal']:
            ints = np.zeros((len(self.kkkmu_list), 2, 2))
            for i in range(len(self.kkkmu_list)):
                kmuargs = self.kkkmu_list[i]
                ints[i] = self.integrand_bs(kmuargs, *args, coordinate=coordinate, unique=unique, mu_opt=mu_opt)
            if method == 'simpson':
                int_func = integrate.simps
            if method == 'trapezoidal':
                int_func = integrate.trapz
            
            ints = ints.reshape(*(self.divs), 2, 2)
            self.ints = ints
            ints = int_func(ints, self.k1_list, axis=0)
            if coordinate == 'cartesian':
                ints = int_func(ints, self.k2_list, axis=0)
                ints = int_func(ints, self.k3_list, axis=0)
            if coordinate == 'child18':
                ints = int_func(ints, self.delta_list, axis=0)
                ints = int_func(ints, self.theta_list, axis=0)
            if coordinate == 'ascending':
                ints = int_func(ints, self.delta1_list, axis=0)
                ints = int_func(ints, self.delta2_list, axis=0)

            if 'RSD' in self.ingredients:
                ints = int_func(ints, self.mu1_list, axis=0)
                ints = int_func(ints, self.mu2_list, axis=0)
            else:
                ints = ints[0, 0]
            return ints

    def integrand_2d(self, args, k1_min=0.01, k1_max=0.2, div_k1=19, z=0, coordinate='child18'):
        """
        args can be k2 k3 or delta theta
        """
        res = 0.0
        dk1 = (k1_max-k1_min)/div_k1
        k1_list = np.linspace(k1_min+dk1/2, k1_max-dk1/2, num=div_k1)
        for k1 in k1_list:
            res += self.integrand_bs((k1, *args, 0., 0.,), z=z, coordinate=coordinate)
        return res*dk1

    def fisher_matrix_bs(self, regions, method='naive', addprior=False, tol=1e-4, rtol=1e-4, unique=True, verbose=False, k_max_bi=2333):
        """
        todos:  - test this method
                - the upper bound of k1 is probably too small
                - check higher orders; or SPT alternative
        """
        """
        integration methods: naive, monte_carlo, simpson, trapezoidal
        divideby: num, step
        """
        fisher_bs_list = np.zeros((len(self.zmid_list), 2, 2))

        self.evaluation_count = 0
        self.sampling_count = 0
        self.k_max_bi = k_max_bi
        
        for z in self.zmid_list:
            fisher_temp = np.zeros((2,2))
            if z+self.dz/2 <= self.z_max_int:
                v = self.survey_volume(self.f_sky, z-self.dz/2, z+self.dz/2)
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
                
                
                if 'RSD' not in self.ingredients:
                    bounds[-1] = [0, 0]
                    bounds[-2] = [0, 0]
                    self.divs[-1] = 1
                    self.divs[-2] = 1
                    self.dds[-1] = 1.0
                    self.dds[-2] = 1.0
                
                if subregion['coordinate'] == 'cartesian':
                    keys = ['k1', 'k2', 'k3', 'mu1', 'mu2']
                elif subregion['coordinate'] == 'child18':
                    keys = ['k1', 'delta', 'theta', 'mu1', 'mu2']
                elif subregion['coordinate'] == 'ascending':
                    keys = ['k1', 'delta1', 'delta2', 'mu1', 'mu2']
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
                    setattr(self, key+'_list', temp_list)

                if subregion['coordinate'] == 'cartesian':
                    kkkmu_list = list(itertools.product(self.k1_list, self.k2_list, self.k3_list, self.mu1_list, self.mu2_list))
                    
                if subregion['coordinate'] == 'child18':
                    kkkmu_list = list(itertools.product(self.k1_list, self.delta_list, self.theta_list, self.mu1_list, self.mu2_list))

                if subregion['coordinate'] == 'ascending':
                    kkkmu_list = list(itertools.product(self.k1_list, self.delta1_list, self.delta2_list, self.mu1_list, self.mu2_list))

                self.kkkmu_list = kkkmu_list
                fisher_temp += v/(8*np.pi**4)*self.naive_integration_bs(args=(z,), coordinate=subregion['coordinate'], method=method, unique=unique, mu_opt=subregion['mu_opt'])
                self.sampling_count += np.prod(self.divs)

            fisher_bs_list[self.zmid_list==z] = fisher_temp
            if verbose is True:
                print('%.3f'%z, fisher_temp.flatten())
        
        self.fisher_bs_list = np.array(fisher_bs_list)
        self.fisher_bs = np.sum(fisher_bs_list, axis=0)
        if addprior == True:
            self.fisher_bs[0,0] += 1/self.alpha_prior['stdev']**2
            self.fisher_bs[1,1] += 1/self.beta_prior['stdev']**2
        fisher_bs_inv = np.linalg.inv(self.fisher_bs)
        self.alpha_stdev_bs = np.sqrt(fisher_bs_inv[0,0])
        self.beta_stdev_bs = np.sqrt(fisher_bs_inv[1,1])
        return self.fisher_bs


    def add_prior(self, fisher):
        fisher[0,0] += 1/self.alpha_prior['stdev']**2
        fisher[1,1] += 1/self.beta_prior['stdev']**2
        fisher_inv = np.linalg.inv(fisher)
        alpha_stdev = np.sqrt(fisher_inv[0,0])
        beta_stdev = np.sqrt(fisher_inv[1,1])
        return fisher, alpha_stdev, beta_stdev


