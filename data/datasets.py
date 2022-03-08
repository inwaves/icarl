import numpy as np
from torch.utils.data import Dataset
from scipy.stats import norm as normal
from models.featuriser import Featuriser


class SyntheticDataset(Dataset):

    def __init__(self, sigma, num_samples, featuriser_type,
                 data_dim):
        self.sigma = sigma
        self.num_samples = num_samples
        self.Z, self.Y = self.samples(sigma, num_samples)
        g = Featuriser(featuriser_type, self.Z.shape[1], data_dim)
        self.X = g(self.Z)
        self.len = self.X.shape[0]
        self.latent_dim = self.Z.shape[1]
        self.data_dim = self.X.shape[1]
        self.aux_dim = self.Y.shape[1]

    def samples(self, sigma, num_samples):
        Z1_samples = normal(loc=0, scale=sigma[0]) \
            .rvs(size=num_samples)
        Y_samples = normal(loc=0, scale=sigma[1]) \
            .rvs(size=num_samples) + Z1_samples
        Z2_samples = normal(loc=0, scale=sigma[2]) \
             .rvs(size=num_samples) + Y_samples

        Z = np.array(list(zip(Z1_samples, Z2_samples)))
        Y_samples = Y_samples.reshape((Y_samples.shape[0], 1))
        return Z, Y_samples

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.Z[index]

    def __len__(self):
        return self.len

    def get_dims(self):
        return self.data_dim, self.aux_dim, self.latent_dim