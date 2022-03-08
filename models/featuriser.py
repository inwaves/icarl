import numpy as np
# import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.functional as F


class Featuriser:
    def __init__(self, type, input_dim, output_dim, hidden_dim=10):
        self.type = type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def __call__(self, X):
        if self.type == 'identity':
            return self.identity(X)
        elif self.type == 'linear':
            return self.linear(X)
        elif self.type == 'nonlinear':
            return self.nonlinear(X)
        else:
            raise ValueError('Unknown featuriser type')

    def identity(self, X):
        return X

    def linear(self, X):
        S = np.random.rand(self.input_dim, self.output_dim)
        return X.dot(S)

    def nonlinear(self, X):
        g = MLP(self.input_dim, self.hidden_dim, self.output_dim)
        X = torch.tensor(X, dtype=torch.float32)
        return g(X).detach().numpy()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.save_hyperparameters()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.out(self.fc1(x))

    # def training_step(self, batch, batch_idx):
    #     idx, targets = batch[:, 0], batch[:, 1]
    #     out = self.forwarc(idx)
    #     loss = F.mse_loss(out, targets)
    #
    #     self.log("train_loss", loss)
    #     return loss
    #
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-5)
