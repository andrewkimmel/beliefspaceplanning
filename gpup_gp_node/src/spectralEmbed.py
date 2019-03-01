# coding: utf-8

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding


class spectralEmbed():


    def __init__(self, embedding_dim, k=100):
        self.embedding_dim = embedding_dim
        self.k = k

        self.embedding = SpectralEmbedding(n_components=self.embedding_dim) 


    def ReducedClosestSetIndices(self, sa, X, k_manifold=10):

        X_spectral = self.embedding.fit_transform(X) 
        sa_r = X_spectral[0,:]

        nbrs = NearestNeighbors(n_neighbors=k_manifold, algorithm='auto').fit(X_spectral)
        distances, indices = nbrs.kneighbors(sa_r.reshape(1,-1))

        return indices


