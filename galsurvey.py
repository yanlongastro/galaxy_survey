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
import functools

def f_phase(k): 
    return 0.227/(1+(0.0324/k)**0.872)

class cosmology:
    """
    cosmology parameters: Omega_m, Omega_b, Omega_L, h, s_f
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


class ps_interpolation:
    """
    Intepolates the power spectrum from data.
    ps, ps_nw: txt files
    """
    def __init__(self, ps_file, ps_nw_file=None):
        pdata = np.loadtxt(ps_file)
        pnwdata = np.loadtxt(ps_nw_file)
        pdata_log = np.log10(pdata)
        p_func_loglog = interpolate.interp1d(pdata_log[:,0], pdata_log[:,1], fill_value="extrapolate")
        self.matter_power_spectrum = lambda x: 10**(p_func_loglog(np.log10(x)))
        pnwdata_log = np.log10(pnwdata)
        pnw_func_loglog = interpolate.interp1d(pnwdata_log[:,0], pnwdata_log[:,1], fill_value="extrapolate")
        self.matter_power_spectrum_no_wiggle = lambda x: 10**(pnw_func_loglog(np.log10(x)))
        oscdata = np.transpose([pdata[:,0], pdata[:,1]/pnwdata[:,1]-1])
        self.oscillation_part = interpolate.interp1d(oscdata[:,0], oscdata[:,1], fill_value="extrapolate")
    
    
class survey:
    """
    cosmology: cosmology class
    ps: ps_interpolation class
    survey geometrics (dict): f_sky, N_g, z_min, z_max, dz, ng_z_list ([zmid_list, ng_list])
    survey parameters (dict): Sigma_0, reconstruction_rate, b_0, survey_type
    ingredients (list): 'RSD', 'damping', 'FOG', 'galactic_bias'
    priors (dict of dict): alpha_prior, beta_prior ([mean, stdev])
    """
    def __init__(self, cosmo, ps, survey_geometrics, survey_parameters, ingredients, priors):
        self.cosmo = cosmo
        self.ps = ps
        for key in survey_geometrics:
            setattr(self, key, survey_geometrics[key])
        for key in survey_parameters:
            setattr(self, key, survey_parameters[key])
        self.ingredients = ingredients
        for key in priors:
            setattr(self, key, priors[key])

        self.V_tot = self.survey_volume(self.f_sky, self.z_min, self.z_max)
        if hasattr(self, 'ng_z_list'):
            self.ng = interpolate.interp1d(ng_z_list[:,0], ng_z_list[:,1], fill_value="extrapolate")
            self.zmid_list = ng_z_list[:,0]
        else:
            self.ng = lambda x: self.N_g/self.V_tot
            number_z = int(np.ceil((self.z_max-self.z_min)/self.dz))
            self.zmid_list = np.linspace(self.z_min+self.dz/2, self.z_max-self.dz/2, num=number_z)
        
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
        bias = self.cosmo.linear_growth_factor(0)/self.cosmo.linear_growth_factor(z)*self.b_0
        return bias

    def damping_factor(self, k, mu=0.0, z=0.0):
        if 'damping' not in self.ingredients:
            return 1.0
        Sigma_vertical = self.Sigma_0*self.cosmo.linear_growth_factor(z)*self.reconstruction_rate
        Sigma_parallel = self.Sigma_0*self.cosmo.linear_growth_factor(z)*(1+self.cosmo.linear_growth_rate(z))*self.reconstruction_rate
        damping = np.exp(-0.5*k**2* (Sigma_vertical**2+mu**2*(Sigma_parallel**2-Sigma_vertical**2)))
        return damping

    def rsd_factor(self, z, mu=0.0):
        if 'RSD' not in self.ingredients:
            return 1.0
        rsd = self.galactic_bias(z) + self.cosmo.linear_growth_rate(z)*mu**2
        return rsd**2

    def oscillation_part(self, k, mu=0.0, z=0.0, damp=True):
        osc = self.ps.oscillation_part(k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f)
        osc *= (self.damping_factor(k, mu, z) if damp==True else 1.0)
        return osc
        
    def power_spectrum(self, k, mu=0.0, z=0.0):
        p = self.ps.matter_power_spectrum_no_wiggle(k)
        p *= 1 + self.oscillation_part(k, mu, z)
        p *= self.rsd_factor(z, mu)
        p *= (self.cosmo.linear_growth_factor(z)/self.cosmo.linear_growth_factor(0))**2
        return p

    def power_spectrum_derivative(self, k, mu=0.0, z=0.0):
        p = self.ps.matter_power_spectrum_no_wiggle(k)
        k_t = k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f
        dodk = misc.derivative(self.ps.oscillation_part, k_t, dx=1e-6)
        dpdk = p*dodk
        dpdk *= self.rsd_factor(z, mu)
        dpdk *= (self.cosmo.linear_growth_factor(z)/self.cosmo.linear_growth_factor(0))**2
        dpdk *= self.damping_factor(k, mu, z)
        dpd_alpha = dpdk*(-k/self.alpha_prior['mean']**2)
        dpd_beta = dpdk*(f_phase(k)/self.cosmo.s_f)
        return np.array([dpd_alpha, dpd_beta])

    @functools.lru_cache(maxsize=None)
    def integrand_ps(self, k, mu, z, i, j):
        dp = self.power_spectrum_derivative(k, mu, z)
        integrand_dp = dp[i]*dp[j]
        integrand_cov = 1/(self.power_spectrum(k, mu, z)+1/self.ng(z))**2
        integrand = integrand_dp*integrand_cov* k**2
        return integrand

    def fisher_matrix_ps(self, regions, addprior=True):
        """
        input: 1d numpy arrays
        refer __init__():
            a) ng_z_list defined: ng = ng(z)
            b) -- not defined: treat ng = Ng/V, and the integral can be 'analytical'
        """
        fisher_ps_list = np.zeros((len(self.zmid_list), 2, 2))
        fisher_temp = np.zeros((2,2))
        for z in self.zmid_list:
            v = self.survey_volume(self.f_sky, z-self.dz, z+self.dz)
            #print(z)
            for i in range(2):
                for j in range(i, 2):
                    if 'RSD' not in self.ingredients:
                        for subregion in regions:
                            k_min = subregion['k_min']
                            k_max = subregion['k_max']
                            fisher_temp[i,j] += v/(4*np.pi**2)*integrate.quad(self.integrand_ps, k_min, k_max, args=(0, z, i, j), limit=1000, epsrel=1e-5, epsabs=1e-4)[0]
                    else:
                        for subregion in regions:
                            k_min = subregion['k_min']
                            k_max = subregion['k_max']
                            mu_min = subregion['mu_min']
                            mu_max = subregion['mu_max']
                            fisher_temp[i,j] += v/(8*np.pi**2)*integrate.dblquad(self.integrand_ps, mu_min, mu_max, lambda mu: k_min, lambda mu: k_max, args=(z, i, j), epsabs=1e-4, epsrel=1e-5)[0]
            fisher_temp[1, 0] = fisher_temp[0, 1]
            #print(fisher_temp)
            fisher_ps_list[self.zmid_list==z] = fisher_temp

        self.fisher_ps_list = np.array(fisher_ps_list)
        self.fisher_ps = np.sum(fisher_ps_list, axis=0)
        if addprior == True:
            self.fisher_ps[0,0] += 1/self.alpha_prior['stdev']**2
            self.fisher_ps[1,1] += 1/self.beta_prior['stdev']**2
        fisher_ps_inv = np.linalg.inv(self.fisher_ps)
        self.alpha_stdev = np.sqrt(fisher_ps_inv[0,0])
        self.beta_stdev = np.sqrt(fisher_ps_inv[1,1])
        return self.fisher_ps


        


    def bipsectrum(self, k_1, k_2, k_3):
        pass

    def bispectrum_tf(self, k_1, delta, theta):
        pass