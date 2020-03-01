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
import itertools


def f_phase(k): 
    return 0.227/(1+(0.0324/k)**0.872)

def k1_tf(k1, delta, theta):
    return k1

def k2_tf(k1, delta, theta):
    return k1+delta

def k3_tf(k1, delta, theta):
    return np.sqrt(k1**2+(k1+delta)**2+2*k1*(k1+delta)*np.cos(theta))

def beta(x):
    if x==1 or x==-1:
        return .5
    elif -1<x<1:
        return 1
    else:
        return 0

def cost(k1, k2, k3):
    return (k3**2-k1**2-k2**2)/(2*k1*k2)

def f_kernal(k1, k2, cos12):
    return 5./7.+.5*(k1/k2+k2/k1)*cos12+2./7.*cos12**2

def s123(k1, k2, k3):
    if k1==k2==k3:
        s123 = 6
    elif k1==k2 or k2==k3 or k3==k1:
        s123 = 2
    else:
        s123 = 1
    return s123
    

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
    attribute later >cosmo: cosmology class
    attribute later>ps: ps_interpolation class
    survey geometrics (dict): f_sky, N_g, z_min, z_max, dz, ng_z_list ([zmid_list, ng_list])
    survey parameters (dict): Sigma_0, reconstruction_rate, b_0, survey_type
    ingredients (list): 'RSD', 'damping', 'FOG', 'galactic_bias'
    priors (dict of dict): alpha_prior, beta_prior ([mean, stdev])

    todos:  - add a fiducial cosmology
            - add survey_type
            - add priors to power/bispectrum calculations: should only affect oscillation parts
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

    #def get_ready(self):
        if hasattr(self, 'ng_z_list'):
            self.z_min = self.ng_z_list[0,0] - self.dz/2
            self.z_max = self.ng_z_list[-1,0] + self.dz/2
        self.V_tot = self.survey_volume(self.f_sky, self.z_min, self.z_max)

        if self.survey_type == 'spectroscopic':
            if hasattr(self, 'ng_z_list'):
                self.ng = interpolate.interp1d(self.ng_z_list[:,0], self.ng_z_list[:,1], fill_value="extrapolate")
                self.zmid_list = self.ng_z_list[:,0]
            else:
                self.ng = lambda x: self.N_g/self.V_tot
                number_z = int(np.ceil((self.z_max-self.z_min)/self.dz))
                self.zmid_list = np.linspace(self.z_min+self.dz/2, self.z_max-self.dz/2, num=number_z)
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

    def rsd_factor(self, z, mu=0.0):
        if 'RSD' not in self.ingredients:
            return 1.0
        rsd = self.galactic_bias(z) + self.cosmo.linear_growth_rate(z)*mu**2
        return rsd**2

    def oscillation_part(self, k, mu=0.0, z=0.0, damp=True, priors=True):
        if priors == True:
            k = k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f
        osc = self.ps.oscillation_part(k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f)
        osc *= (self.damping_factor(k, mu, z) if damp==True else 1.0)
        return osc
        
    def power_spectrum(self, k, mu=0.0, z=0.0, nw=False):
        p = self.ps.matter_power_spectrum_no_wiggle(k)
        if nw==False:
            p *= 1 + self.oscillation_part(k, mu, z)
        p *= self.rsd_factor(z, mu)
        p *= (self.cosmo.linear_growth_factor(z)/self.cosmo.D0)**2
        return p

    def power_spectrum_derivative(self, k, mu=0.0, z=0.0):
        k_t = k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f
        dodk = misc.derivative(self.ps.oscillation_part, k_t, dx=1e-6)
        p = self.ps.matter_power_spectrum_no_wiggle(k)
        dpdk = p*dodk
        dpdk *= self.rsd_factor(z, mu)
        dpdk *= (self.cosmo.linear_growth_factor(z)/self.cosmo.D0)**2
        dpdk *= self.damping_factor(k, mu, z)
        dpd_alpha = dpdk*(-k/self.alpha_prior['mean']**2)
        dpd_beta = dpdk*(f_phase(k)/self.cosmo.s_f)
        return np.array([dpd_alpha, dpd_beta])

    @functools.lru_cache(maxsize=None)
    def integrand_ps(self, k, mu, z, i, j, simplify=False):
        if simplify is True:
            k_t = k/self.alpha_prior['mean'] + (self.beta_prior['mean']-1)*f_phase(k)/self.cosmo.s_f
            dodk = misc.derivative(self.ps.oscillation_part, k_t, dx=1e-6)
            damp = self.damping_factor(k, mu, z)
            dodk *= damp
            dod_alpha = dodk*(-k/self.alpha_prior['mean']**2)
            dod_beta = dodk*(f_phase(k)/self.cosmo.s_f)
            do = np.array([dod_alpha, dod_beta])
            integrand_do = do[i]*do[j]
            integrand_cov = 1/(1+self.ps.oscillation_part(k)*damp)**2/(1+1/(self.ng(z)*self.power_spectrum(k, mu, z)))**2
            integrand = integrand_do*integrand_cov* k**2
            return integrand
        dp = self.power_spectrum_derivative(k, mu, z)
        integrand_dp = dp[i]*dp[j]
        integrand_cov = 1/(self.power_spectrum(k, mu, z)+1/self.ng(z))**2
        integrand = integrand_dp*integrand_cov* k**2
        return integrand


    def naive_integration_ps(self, args):
        res = 0
        for kmu in self.kmu_list:
            res += self.integrand_ps(*kmu, *args)
        res *= self.dk*self.dmu
        return res

    def fisher_matrix_ps(self, regions, addprior=True, tol=1e-4, rtol=1e-4, div_k=0, div_mu=0):
        """
        input: 1d numpy arrays
        refer __init__():
            a) ng_z_list defined: ng = ng(z)
            b) -- not defined: treat ng = Ng/V
        """

        fisher_ps_list = np.zeros((len(self.zmid_list), 2, 2))
        fisher_temp = np.zeros((2,2))
        for z in self.zmid_list:
            v = self.survey_volume(self.f_sky, z-self.dz/2, z+self.dz/2)
            #print(z)
            for i in range(2):
                for j in range(i, 2):
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
                                fisher_temp[i,j] = v/(4*np.pi**2)*self.naive_integration_ps(args=(z, i, j))
                            else:
                                fisher_temp[i,j] += v/(4*np.pi**2)*integrate.quad(self.integrand_ps, k_min, k_max, args=(0, z, i, j), limit=1000, epsrel=rtol, epsabs=tol)[0]
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
                                fisher_temp[i,j] = v/(8*np.pi**2)*self.naive_integration_ps(args=(z, i, j))
                            else:
                                fisher_temp[i,j] += v/(8*np.pi**2)*integrate.dblquad(self.integrand_ps, mu_min, mu_max, lambda mu: k_min, lambda mu: k_max, args=(z, i, j), epsabs=tol, epsrel=rtol)[0]
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


    def bispectrum(self, kargs, mu=0, z=0, coordinate='cartesian', nw=False):
        """
        todos: derive bispectrum in galsurvey, currently the functions should use with no ingredients
        """
        if coordinate == 'cartesian':
            k_1, k_2, k_3 = kargs
        elif coordinate == 'child18':
            k_1, k_2, k_3 = k1_tf(*kargs), k2_tf(*kargs), k3_tf(*kargs)
        cos12, cos23, cos31 = cost(k_1, k_2, k_3), cost(k_2, k_3, k_1), cost(k_3, k_1, k_2)
        f12, f23, f31 = f_kernal(k_1, k_2, cos12), f_kernal(k_2, k_3, cos23), f_kernal(k_3, k_1, cos31)
        p1, p2, p3 = self.power_spectrum(k_1, mu=mu, z=z, nw=nw), self.power_spectrum(k_2, mu=mu, z=z, nw=nw), self.power_spectrum(k_3, mu=mu, z=z, nw=nw)
        return 2*(p1*p2*f12+p2*p3*f23+p3*p1*f31)

    def bispectrum_derivative(self, kargs, mu=0, z=0, coordinate='cartesian'):
        """
        will give different results depending on the coordinate
        """
        if coordinate =='cartesian':
            k_1, k_2, k_3 = kargs
            cos12, cos23, cos31 = cost(k_1, k_2, k_3), cost(k_2, k_3, k_1), cost(k_3, k_1, k_2)
            f12, f23, f31 = f_kernal(k_1, k_2, cos12), f_kernal(k_2, k_3, cos23), f_kernal(k_3, k_1, cos31)
            p1, p2, p3 = self.power_spectrum(k_1, mu=mu, z=z), self.power_spectrum(k_2, mu=mu, z=z), self.power_spectrum(k_3, mu=mu, z=z)
            dp1, dp2, dp3 = self.power_spectrum_derivative(k_1, mu=mu, z=z), self.power_spectrum_derivative(k_2, mu=mu, z=z), self.power_spectrum_derivative(k_3, mu=mu, z=z)
            res = 2*((dp1*p2+p1*dp2)*f12 +(dp2*p3+p2*dp3)*f23+(dp3*p1+p3*dp1)*f31)
            return res
        elif coordinate =='child18':
            pass



    def R_bi(self, kargs, mu=0, z=0):
        return self.bispectrum(kargs, mu=mu, z=z, coordinate='child18')/self.bispectrum(kargs, mu=mu, z=z, coordinate='child18', nw=True)

    def A_bi(self, delta, theta, k1_min=0.01, k1_max=0.2, div_k1=10000, mu=0, z=0):
        k1sample = np.linspace(k1_min, k1_max, num=div_k1)
        Rsample = self.R_bi(kargs=(k1sample, delta, theta), mu=mu, z=z)
        Rmean = np.mean(Rsample)
        #Rmean = integrate.quad(R, kmin_bi, kmax_bi, limit=100)[0]/(kmax_bi-kmin_bi)
        #A2 = lambda x: (R(x)-Rmean)**2
        #A2_val, err = integrate.quad(A2, kmin_bi, kmax_bi, limit=1000)
        A2sample = (Rsample-Rmean)**2
        A2 = np.mean(A2sample)
        return np.sqrt(A2)

    def integrand_bs(self, kargs, mu, z, i, j, coordinate='cartesian', simplify=False):
        if coordinate == 'cartesian':
            db = self.bispectrum_derivative(kargs, mu=mu, z=z, coordinate=coordinate)
            integrand_db = db[i]*db[j]
            k1, k2, k3 = kargs
            p1, p2, p3 = self.power_spectrum(k1, mu=mu, z=z), self.power_spectrum(k2, mu=mu, z=z), self.power_spectrum(k3, mu=mu, z=z)
            integrand_cov = k1*k2*k3*beta(cost(*kargs))/s123(*kargs)/(p1*p2*p3)
            integrand = integrand_db*integrand_cov
            return integrand
        elif coordinate == 'child18':
            pass

    def naive_integration_bs(self, args, coordinate='cartesian'):
        res = 0
        for kmuargs in self.kkkmu_list:
            res += self.integrand_bs(*kmuargs, *args, coordinate=coordinate)
        if coordinate == 'cartesian':
            res *= self.dk1*self.dk2*self.dk3
            return res
        elif coordinate == 'child18':
            pass

    def fisher_matrix_bs(self, regions, coordinate='cartesian', addprior=True, tol=1e-4, rtol=1e-4, div_k1=0, div_k2=0, div_k3=0, div_mu=0):
        """
        todos:  - test this method
                - add RSD
                - add child18 coordinate
        """
        fisher_bs_list = np.zeros((len(self.zmid_list), 2, 2))
        fisher_temp = np.zeros((2,2))
        for z in self.zmid_list:
            v = self.survey_volume(self.f_sky, z-self.dz/2, z+self.dz/2)
            #print(z)
            for i in range(2):
                for j in range(i, 2):
                    if 'RSD' not in self.ingredients:
                        for subregion in regions:
                            k1_min = subregion['k1_min']
                            k1_max = subregion['k1_max']
                            k2_min = subregion['k2_min']
                            k2_max = subregion['k2_max']
                            k3_min = subregion['k3_min']
                            k3_max = subregion['k3_max']
                            if div_k1 !=0 and div_k2 !=0 and div_k3 !=0:
                                self.dk1 = dk1 = (k1_max-k1_min)/div_k1
                                self.dk2 = dk2 = (k2_max-k2_min)/div_k2
                                self.dk3 = dk3 = (k3_max-k3_min)/div_k3
                                self.dmu = 1.0
                                k1_list = np.linspace(k1_min+dk1/2, k1_max-dk1/2, num=div_k1)
                                k2_list = np.linspace(k2_min+dk2/2, k2_max-dk2/2, num=div_k2)
                                k3_list = np.linspace(k3_min+dk3/2, k3_max-dk3/2, num=div_k3)
                                mu_list = np.array([0.0])
                                self.kkkmu_list = list(itertools.product(k1_list, k2_list, k3_list, mu_list))
                                fisher_temp[i,j] = v/(np.pi)*self.naive_integration_bs(args=(z, i, j), coordinate=coordinate)
                            else:
                                pass
                    else:
                        pass

            fisher_temp[1, 0] = fisher_temp[0, 1]
            fisher_bs_list[self.zmid_list==z] = fisher_temp
        
        self.fisher_bs_list = np.array(fisher_bs_list)
        self.fisher_bs = np.sum(fisher_bs_list, axis=0)
        if addprior == True:
            self.fisher_bs[0,0] += 1/self.alpha_prior['stdev']**2
            self.fisher_bs[1,1] += 1/self.beta_prior['stdev']**2
        fisher_bs_inv = np.linalg.inv(self.fisher_bs)
        self.alpha_stdev_bs = np.sqrt(fisher_bs_inv[0,0])
        self.beta_stdev_bs = np.sqrt(fisher_bs_inv[1,1])
        return self.fisher_bs
