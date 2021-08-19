import numpy as np
import h5py

class fisher:
    def __init__(self, matrix, keys):
        self.matrix = np.array(matrix)
        self.ndim = matrix.shape[0]
        if len(keys) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Check dimensions of input matrix and keys.')
        self.keys = list(keys)
        
    def merge(self, income):
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
    
    def constraints(self, keys=None):
        sigmas = np.sqrt(np.diag(np.linalg.inv(self.matrix)))
        if keys is None:
            return sigmas
        else:
            cs = []
            for k in keys:
                cs.append(sigmas[self.keys.index(k)])
            return np.array(cs)
    
    def save(self, out):
        if '.txt' in out:
            np.savetxt(out, self.matrix)
        if '.hdf5' in out:
            with h5py.File(out, 'w') as f:
                f.create_dataset('matrix', data=self.matrix)
                dt = h5py.special_dtype(vlen=str)
                keys = np.array(self.keys, dtype=dt) 
                f.create_dataset('keys', data=keys)