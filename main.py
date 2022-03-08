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
import networkx as nx
import matplotlib.pyplot as plt
import cdt
import pandas as pd
import numpy as np


from utils.utils import Logger
from torch.utils.data import DataLoader
from cdt.causality.graph import PC
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci

from data.datasets import SyntheticDataset
from models.ivae import IVAE
from utils.metrics import mean_corr_coef as mcc

cdt.SETTINGS.rpath = '/usr/local/bin/rscript'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_samples = 10000
epochs = 100
hidden_dim = 512
batch_size = 64


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
    σ = [[1, 0, σ3] for σ3 in [0.2, 2, 100]]
    dataset = SyntheticDataset(σ[0], num_samples, "linear", 10)
    data_dim, aux_dim, latent_dim = dataset.get_dims()
    print(f"Dimensions: {data_dim} x {aux_dim} -> {latent_dim}")

    # Train an iVAE to recover the latent variables Z, given X, E, Y.
    model = IVAE(latent_dim, data_dim, aux_dim, hidden_dim)
    optimiser = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.1, patience=4, verbose=True)

    train_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    perf, elbo, latents = train(model, train_dataloader, optimiser, scheduler, epochs, device)
    print(f"perf: {perf}")
    print(f"Finished training the iVAE\n==================================================")
    Z_est = np.concatenate(latents[-1])

    print(f"Shape of Y: {dataset.Y.shape}, of Z: {dataset.Z.shape}, of Z_est: {Z_est.shape}")

    # Run the PC algorithm on the estimated latents.
    df = pd.DataFrame(np.concatenate((Z_est, dataset.Y), axis=1))
    cg1 = pc(Z_est, indep_test=kci)
    cg2 = pc(np.concatenate((Z_est, dataset.Y), axis=1), indep_test=kci)
    cg1.to_nx_graph()
    cg1.draw_nx_graph(skel=False)
    # cg2.to_nx_graph()
    # cg2.draw_nx_graph(skel=False)
    # lasso = cdt.independence.graph.Glasso()
    # skeleton = lasso.predict(df)
    # dag = PC()
    # output = dag.predict(df, skeleton)
    print(f"Finished running the PC algorithm\n==================================================")
    # nx.draw_networkx(output, with_labels=True)
    # print(f"Output: {output}")
    plt.show()

    # Conditional independence test between the recovered latents conditioned on the environment.
    # And again between the latents and the target variable. The results are the same if the latents do not cause Y.
    # kci(Z_est, X=0, Y=1)
    # kci(np.concatenate((Z_est, dataset.Y), axis=1), X=0, Y=1, condition_set=2)
