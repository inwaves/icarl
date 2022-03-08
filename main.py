"""
End-to-end invariant causal representation learning (ICARL) comprises:
    1. Given a dataset comprising observations X, U and a target variable Y,
    use an identifiable VAE to estimate a latent representation of X.
    2a. Given an estimated latent representation, use the PC algorithm to
    recover a skeleton graph of the causal structure of Y.
    2b. Given the skeleton graph, use kernel causal independence (KCI) tests
    to find the parents of Y in the graph, Pa(Y).
    3. Given Pa(Y), learn an invariant representation Φ(Pa(Y)) and a predictor
     w which is optimal across environments.

In theory, you could stop at step 2b, if all you were interested in was the
sources/factors of variation. But the main goal of ICARL is out-of-distribution
generalisation, this is why we train the predictor w.
"""
import torch
import torch.optim as optim
from utils.utils import Logger
from torch.utils.data import DataLoader

from data.datasets import SyntheticDataset
from models.ivae import IVAE
from utils.metrics import mean_corr_coef as mcc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_samples = 1000
epochs = 1000
hidden_dim = 100
batch_size = 128


def train(model, train_dataloader, optimiser, scheduler, epochs, device):
    """
    Train the model for a number of epochs.
    """
    logger = Logger(name="VAE log")
    logger.add("elbo")
    logger.add("perf")

    model.train()
    for epoch in range(epochs):
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

    return perf, elbo, Z_est


if __name__ == '__main__':
    σ = [[1, 0, σ3] for σ3 in [0.2, 2, 100]]
    dataset = SyntheticDataset(σ[0], num_samples, "linear", 10)
    data_dim, aux_dim, latent_dim = dataset.get_dims()

    # Train an iVAE to recover the latent variables Z, given X, E, Y.
    model = IVAE(latent_dim, data_dim, aux_dim, hidden_dim)
    optimiser = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.1, patience=4, verbose=True)

    train_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    perf, elbo, Z_est = train(model, train_dataloader, optimiser, scheduler, epochs, device)
    print(f"Estimated Z: {Z_est}, actual Z: {dataset.Z}")
    print(f"perf: {perf}")
