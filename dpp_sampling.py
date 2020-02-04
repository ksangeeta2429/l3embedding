import h5py
import numpy as np
from numpy.random import rand, randn, RandomState
from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear

feats_path = '/beegfs/work/sonyc/features/openl3/2017/sonycnode-b827eb132382.sonyc_features_openl3.h5'
blob = h5py.File(feats_path)
feats = blob['openl3']['openl3']
feats = feats.reshape(-1, 512)[:100000,:]

L = feats.dot(feats.T)
DPP = FiniteDPP('likelihood', **{'L': L})

k = 1000
#for _ in range(10):
DPP.sample_mcmc_k_dpp(size=k)
#DPP.sample_mcmc_k_dpp(size=k)

print(len(DPP.list_of_samples))
print(DPP.list_of_samples)