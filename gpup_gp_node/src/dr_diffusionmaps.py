#!/usr/bin/env python

import numpy as np
import time
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors


class DiffusionMap():

    def __init__(self, sigma=10, embedding_dim=2):

        self.sigma2 = sigma**2
        self.dim = embedding_dim

    def fit_transform(self, X):

        N = X.shape[0]
        data = X

        knn = int(np.ceil(0.03*float(N)))

        m = data.shape[0]
        dt = squareform(pdist(data))
        srtdDt = np.sort(dt, axis=0)
        srtdIdx = np.argsort(dt, axis=0)
        dt = srtdDt[:knn+1,:]
        nidx = srtdIdx[:knn+1,:]

        tempW = np.exp(-dt**2 / self.sigma2)
        
        i = np.tile(np.array(range(m)), (knn+1, 1))
        W = csr_matrix((tempW.reshape((-1,)), (i.reshape((-1,)), nidx.reshape((-1,)))), shape=(m, m))#.toarray()
        Wt = csr_matrix((tempW.reshape((-1,)), (nidx.reshape((-1,)), i.reshape((-1,)))), shape=(m, m))#.toarray()
        BisBigger = W > Wt
        W = Wt - Wt.multiply(BisBigger) + W.multiply(BisBigger) # =max(W, W^T)
        
        ld = csr_matrix(np.diag((1./np.sqrt(W.sum(axis=1))).A.reshape((-1,))), shape=(m, m))
        D0 = ld * W * ld
        D0t = csr_matrix(D0.A.T, shape=(m, m))
        BisBigger = D0 > D0t 
        D0 = D0t - D0t.multiply(BisBigger) + D0.multiply(BisBigger) # =max(D0, D0^T)

        # D, V = np.linalg.eig(np.array(D0.A))#, which='LR')#, k=2+0*np.min(m-1, 10))#)
        D, V = eigs(D0, np.min((m-2, 10)), which='LR')#, #)
        
        idx = np.flipud(np.argsort(D.real))
        lambdas = D.real[idx]
        v = V.real[:,idx[:self.dim]]

        return v, lambdas

    def ReducedClosestSetIndices(self, sa, X, k_manifold=10):

        V, lam = self.fit_transform(X)
        sa_r = V[0,:]
        # V = V[1:,:]

        nbrs = NearestNeighbors(n_neighbors=k_manifold, algorithm='auto').fit(V)
        distances, indices = nbrs.kneighbors(sa_r.reshape(1,-1))

        return indices



