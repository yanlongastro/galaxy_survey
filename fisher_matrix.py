import numpy as np
import h5py
from getdist import plots, MCSamples
import copy

class fisher:
    '''
    Fisher matrix manipulation
    '''
    def __init__(self, matrix, keys):
        self.matrix = np.array(matrix)
        self.ndim = matrix.shape[0]
        if len(keys) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Check dimensions of input matrix and keys.')
        self.keys = list(keys)

    def element(self, k1, k2=None):
        if k2 is None:
            k2 = k1
        e1 = self.keys.index(k1)
        e2 = self.keys.index(k2)
        return self.matrix[e1, e2]
        
        
    def merge(self, income=None):
        if income == None:
            return self
        out_keys = list(set(self.keys).union(set(income.keys)))
        out_ndim = len(out_keys)
        out_matrix = np.zeros((out_ndim, out_ndim))
        for i in range(out_ndim):
            for j in range(out_ndim):
                k1, k2 = out_keys[i], out_keys[j]
                for keys, matrix in zip([self.keys, income.keys], [self.matrix, income.matrix]):
                    if k1 in keys and k2 in keys:
                        e1 = keys.index(k1)
                        e2 = keys.index(k2)
                        out_matrix[i, j] += matrix[e1, e2]
        return fisher(out_matrix, out_keys)

    def slice(self, keys=None, exclude=False):
        if keys is None or len(keys)==0:
            matrix = self.matrix
            keys = self.keys
        else:
            if exclude:
                keys = list(set(self.keys).difference(set(keys)))
            ndim = len(keys)
            matrix = np.zeros((ndim, ndim))
            for i in range(ndim):
                for j in range(ndim):
                    k1, k2 = keys[i], keys[j]
                    e1 = self.keys.index(k1)
                    e2 = self.keys.index(k2)
                    matrix[i, j] = self.matrix[e1, e2]
        return fisher(matrix, keys)
    
    def constraints(self, keys=None):
        sliced = self.slice(keys)
        sigmas = np.sqrt(np.diag(np.linalg.inv(sliced.matrix)))
        sigma_dict = {}
        for i in range(sliced.ndim):
            sigma_dict[sliced.keys[i]] = sigmas[i]
        return sigmas, sigma_dict

    def normalize(self, norms={}):
        matrix = copy.deepcopy(self.matrix)
        for k in norms.keys():
            e = self.keys.index(k)
            matrix[e, :] /= norms[k]
            matrix[:, e] /= norms[k]
        return fisher(matrix, self.keys)
    
    def save(self, out):
        if '.txt' in out:
            np.savetxt(out, self.matrix)
        if '.hdf5' in out:
            with h5py.File(out, 'w') as f:
                f.create_dataset('matrix', data=self.matrix)
                dt = h5py.special_dtype(vlen=str)
                keys = np.array(self.keys, dtype=dt) 
                f.create_dataset('keys', data=keys)

    def write_hdf5(self, f):
        f.create_dataset('matrix', data=self.matrix)
        dt = h5py.special_dtype(vlen=str)
        keys = np.array(self.keys, dtype=dt) 
        f.create_dataset('keys', data=keys)


def read_hdf5(group):
        '''
        read hdf5 file, return a new instance
        '''
        if type(group) == str:
            with h5py.File(group, 'r') as group_:
                return read_hdf5(group_)
        else:
            keys = group['keys'][()]
            matrix = group['matrix'][()]
            return fisher(matrix, keys)



def triangle_plot(fishers, keys=None, fisher_labels=None, parameter_labels=None, nsamp=10000, rgs=10, norms={}):
    random_state = np.random.default_rng(rgs)
    if type(fishers) is not list:
        fishers = [fishers]
    if keys is None:
        keys = fishers[0].keys
    if parameter_labels is not None:
        labels = parameter_labels
    else:
        labels = keys
    names = keys
    ndim = len(keys)
    covs = [f.slice(keys).normalize(norms).matrix for f in fishers]
    samples = []
    if fisher_labels is None:
        fisher_labels = ['Matrix-%d'%(i+1) for i in range(len(fishers))]

    for cov, label in zip(covs, fisher_labels):
        cc = np.linalg.inv(cov)
        samps = random_state.multivariate_normal([0]*ndim, cc, size=nsamp)
        samples.append(MCSamples(samples=samps,names=names, labels=labels, label=label))

    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.alpha_filled_add = 0.5
    g.settings.axes_labelsize = 12
    g.triangle_plot(samples, filled=False)