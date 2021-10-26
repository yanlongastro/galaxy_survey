import numpy as np
from tqdm.notebook import tqdm
import scipy.interpolate as interpolate
import scipy.fftpack as fftpack

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams["figure.dpi"] = 100

import galsurveycy as gs


'''
Isolate BAO wiggles from matter power spectrum.
'''

def dewiggle(k, p, n=20, debug=False, ishift_odd = (-3, 20), ishift_even = (-3, 20), istar_shift=-6):
    '''
    Input: k, p, 1d array
    Output: intepolated functions
    '''
    # interp
    kmin, kmax = k.min(), k.max()
    logk = np.log10(k)
    logp = np.log10(p)
    logkp = np.log10(k*p)
    p_loglog = interpolate.CubicSpline(logk, logp)
    kp_loglog = interpolate.CubicSpline(logk, logkp)
    
    # sample
    ks = np.linspace(kmin, kmax, num=2**n)
    logkps = kp_loglog(np.log10(ks))
    
    # dst
    kps_dst = fftpack.dst(logkps, norm='ortho')
    kps_dst_odd = kps_dst[::2]
    kps_dst_even = kps_dst[1::2]
    
    # dst interp
    steps = np.linspace(0, 2**n//2-1, num=2**n//2)
    odd = interpolate.InterpolatedUnivariateSpline(steps, kps_dst_odd)
    kps_dst_odd_pp = (kps_dst_odd[2:]-kps_dst_odd[:-2])
    odd_pp = interpolate.CubicSpline(steps[1:-1], kps_dst_odd_pp)
    
    even = interpolate.CubicSpline(steps, kps_dst_odd)
    kps_dst_even_pp = (kps_dst_even[2:]-kps_dst_even[:-2])
    even_pp = interpolate.CubicSpline(steps[1:-1], kps_dst_even_pp)
    
    steps = np.linspace(1, 2**n//2, num=2**n//2)
    
    a = odd_pp(steps)
    ids = np.arange(len(a))[np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True] ]
    
    
    a = -even_pp(steps)
    ids = np.arange(len(a))[np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True] ]
    istar = ids[0]
    
    a = -a
    ids = np.arange(len(a))[np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True] ]
    ids = ids[ids>istar]
    istar = ids[0]+istar_shift
    
    if debug:
        print(istar)
    
    
    # trim the bump
    steps = np.linspace(0, 2**n//2-1, num=2**n//2)
    y1 = (kps_dst_odd*(steps+1)**2)[(steps<istar+ishift_odd[0])+(steps>istar+ishift_odd[1])]
    y2 = (kps_dst_even*(steps+1)**2)[(steps<istar+ishift_even[0])+(steps>istar+ishift_even[1])]
    x1 = steps[(steps<istar+ishift_odd[0])+(steps>istar+ishift_odd[1])]
    x2 = steps[(steps<istar+ishift_even[0])+(steps>istar+ishift_even[1])]
    
    # plt.scatter(x, y1)
    oddi2 = interpolate.CubicSpline(x1, y1)
    if debug:
        plt.plot(istar+np.array(ishift_odd), [1, 1], c='k')
        plt.plot(steps, kps_dst_odd - oddi2(steps)/(steps+1)**2, label='odd')
        #plt.plot(steps, kps_dst_odd, label='odd')
        #plt.plot(steps, oddi2(steps)/(steps+1)**2, label='odd')
        plt.xlim(0, 50)
        #plt.ylim(0, 200)
    kps_dst_odd_temp = oddi2(steps)/(steps+1)**2
    kps_dst_odd = kps_dst_odd_temp
    
    eveni2 = interpolate.CubicSpline(x2, y2)
    if debug:
        plt.plot(istar+np.array(ishift_even), [2, 2], 'k--')
        plt.plot(steps, kps_dst_even - eveni2(steps)/(steps+1)**2, label='even')
    kps_dst_even_temp = eveni2(steps)/(steps+1)**2
    kps_dst_even = kps_dst_even_temp
    if debug:
        plt.legend()
        plt.show()
    
    
    # idst
    idst = []
    for i in range(2**n//2):
        idst.append(kps_dst_odd[i])
        idst.append(kps_dst_even[i])
    idst = np.array(idst)
    kpp = fftpack.idst(idst, norm='ortho')
    if debug:
        plt.semilogx(ks, kpp)
        plt.show()
    
    nk = len(logk)
    p_loglog = lambda x: gs.two_value_interpolation_c(logk, logp, x, nk)

    nk = len(ks)
    pnw_loglog = interpolate.CubicSpline(np.log10(ks), kpp-np.log10(ks))
    nk = len(logk)
    logpnw = pnw_loglog(logk)
    pnw_loglog = lambda x: gs.two_value_interpolation_c(logk, logpnw, x, nk)

    p = lambda x: 10**p_loglog(np.log10(x))
    pnw = lambda x: 10**pnw_loglog(np.log10(x))
    return p, pnw




def find_peaks_troughs_zeros(k, o, num=6, kmax=10086):
    '''
    Results are limited by the resolution of k. So input dens k's.
    num: number of periods to count.
    '''
    res = []
    i = 0
    for a in [-o, o, np.abs(o)]:
        ids = np.arange(len(a))[np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True] ]
        ids = ids[(ids>0) & (ids<len(o)-1)]
        ks = k[ids]
        ks = ks[ks<kmax]
        if i == 2:
            num = 2*num
        ks = ks[:num]
        res.append(ks)
        i += 1
    
    return tuple(res)

def e(k):
    Am = 0.141
    a = 0.0035
    kap = 5.5
    kk = 0.016
    return 1-Am*np.exp(-a*(k/kk)**kap)

def d(k):
    Am = 0.072
    a = 0.32
    kap = 1.9
    kk = 0.12
    return Am*np.exp(-a*(k/kk)**kap)

def a(k):
    return d(k)*e(k)