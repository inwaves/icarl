"""
What I want to do with these experiments:
    0. Create a principled data split along dimensions as in VISREP.
    1. Apply an iVAE to the dsprites dataset to recover latent factors of variation.
    The idea is to disentangle the 6 factors that dsprites comprises: colour, shape,
    size, rotation, posX, posY.
    2. Train a classifier on the estimated latents to predict... what?
    3. Train an invariant predictor on the estimated latents, à la IRM.

"""
import torch
import torch.optim as optim
import numpy as np
from utils.utils import Logger
from torch.utils.data import DataLoader

from data.datasets import DSpritesDataset
from models.ivae import IVAE
from utils.metrics import mean_corr_coef as mcc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 10
hidden_dim = 64
batch_size = 64

# TODO: get the latent values for all the images in the dataset.
# TODO: split the data into training, validation and test sets.
# TODO: train the iVAE to reconstruct the data by learning latents.
# TODO: given estimated latents, train a classifier to predict the label of the image? What's the label?
# TODO: once stuff works, reimplement the iVAE in pytorch-lightning.


def train(model, train_dataloader, optimiser, scheduler, epochs, device):
    """
    Train the model for a number of epochs.
    """
    logger = Logger(name="VAE log")
    logger.add("elbo")
    logger.add("perf")

    latents = []
    model.train()
    for epoch in range(epochs):
        current_latents = []
        for _, (X, Y, Z) in enumerate(train_dataloader):
            X, Y, Z = X.to(device).float(), Y.to(device).float(), Z.to(device).float()
            optimiser.zero_grad()
            elbo, Z_est = model.elbo(X, Y)
            elbo.mul(-1).backward()  # Because we want to maximise ELBO.
            optimiser.step()

            logger.update("elbo", -elbo.item())

            perf = mcc(Z.numpy(), Z_est.detach().numpy())
            logger.update("perf", perf.item())

            if epoch % 10 == 0:
                logger.log()
                scheduler.step(logger.get_last("elbo"))

            current_latents.append(Z_est.detach().numpy())
        latents.append(current_latents)

    return perf, elbo, latents


if __name__ == '__main__':

    dataset = DSpritesDataset()
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_dim, aux_dim, latent_dim = dataset.get_dims()
    print(f"Dimensions:\n Data: {data_dim}\nAuxiliary: {aux_dim}\nLatent: {latent_dim}")

    # Train an iVAE to recover the latent variables Z, given X, E, Y.
    model = IVAE(latent_dim, data_dim, aux_dim, hidden_dim)
    optimiser = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.1, patience=4, verbose=True)

    train_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    perf, elbo, latents = train(model, train_dataloader, optimiser, scheduler, epochs, device)
    print(f"perf: {perf}")
    print(f"Finished training the iVAE\n==================================================")
    Z_est = np.concatenate(latents[-1])
