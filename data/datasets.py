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


class DspritesDataset(Dataset):

    def __getitem__(self, index):
        return self.imgs[index], self.shape_latent[index]

    def __init__(self):
        data_filepath = "data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        data = np.load(data_filepath)
        self.imgs = data["imgs"]
        self.latents_values = data["latents_values"]
        self.latents_classes = data["latents_classes"]
        self.metadata = {
            'description': 'Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6 disentangled latent factors.This dataset uses 6 latents, controlling the color, shape, scale, rotation and position of a sprite. All possible variations of the latents are present. Ordering along dimension 1 is fixed and can be mapped back to the exact latent values that generated that image.We made sure that the pixel outputs are different. No noise added.',
            'latents_sizes': np.array([1, 3, 6, 40, 32, 32]),
            'latents_names': ('color', 'shape', 'scale', 'orientation', 'posX', 'posY'), 'date': 'April 2017',
            'version': 1, 'title': 'dSprites dataset',
            'latents_possible_values': {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                                          0.16129032, 0.19354839, 0.22580645, 0.25806452, 0.29032258,
                                                          0.32258065, 0.35483871, 0.38709677, 0.41935484, 0.4516129,
                                                          0.48387097, 0.51612903, 0.5483871, 0.58064516, 0.61290323,
                                                          0.64516129, 0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                                          0.80645161, 0.83870968, 0.87096774, 0.90322581, 0.93548387,
                                                          0.96774194, 1.]),
                                        'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                                          0.16129032, 0.19354839, 0.22580645, 0.25806452, 0.29032258,
                                                          0.32258065, 0.35483871, 0.38709677, 0.41935484, 0.4516129,
                                                          0.48387097, 0.51612903, 0.5483871, 0.58064516, 0.61290323,
                                                          0.64516129, 0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                                          0.80645161, 0.83870968, 0.87096774, 0.90322581, 0.93548387,
                                                          0.96774194, 1.]),
                                        'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                                        'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195, 0.64442926,
                                                                 0.80553658, 0.96664389, 1.12775121, 1.28885852,
                                                                 1.44996584,
                                                                 1.61107316, 1.77218047, 1.93328779, 2.0943951,
                                                                 2.25550242,
                                                                 2.41660973, 2.57771705, 2.73882436, 2.89993168,
                                                                 3.061039,
                                                                 3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                                                 3.86657557,
                                                                 4.02768289, 4.1887902, 4.34989752, 4.51100484,
                                                                 4.67211215,
                                                                 4.83321947, 4.99432678, 5.1554341, 5.31654141,
                                                                 5.47764873,
                                                                 5.63875604, 5.79986336, 5.96097068, 6.12207799,
                                                                 6.28318531]), 'shape': np.array([1., 2., 3.]),
                                        'color': np.array([1.])}, 'author': 'lmatthey@google.com'}

        # Define number of values per latents and functions to convert to indices
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                             np.array([1, ])))

        # Recover the latent values for the shape in each image.
        # Get the shape latent value of the entire dataset.
        # Squares: [0, imgs.shape[0]//3)
        # Ellipses: [imgs.shape[0]//3,  2*imgs.shape[0]//3)
        # Hearts: [2*imgs.shape[0]//3, imgs.shape[0])
        one_third = self.imgs.shape[0] // 3
        self.shape_latent = np.concatenate(
            (np.zeros(one_third),
             np.ones(one_third),
             2 * np.ones(one_third),)
        ).reshape((3 * one_third, 1))

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples
