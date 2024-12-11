import logging
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from losses import log_nb_positive

logger = logging.getLogger(__name__)


activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "softmax": nn.Softmax(dim=1),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "prelu": nn.PReLU(),
    "hardswish": nn.Hardswish(),
    "hardsigmoid": nn.Hardsigmoid(),
    "logsigmoid": nn.LogSigmoid(),
    "silu": nn.SiLU(),  # Also called Swish
    "tanhshrink": nn.Tanhshrink(),
    "hardshrink": nn.Hardshrink(),
    "softshrink": nn.Softshrink(),
}


def get_fully_connected_layers(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "relu",
    norm_type: Literal["layer", "batch", "none"] = "batch",
    dropout_prob: float = 0.
) -> nn.Sequential:
    """Construct fully connected layers."""
    layers = []
    for i, size in enumerate(hidden_dims):
        layers.append(nn.Linear(input_dim, size))
        if activation in activations:
            layers.append(activations[activation])
        if norm_type == "layer":
            layers.append(nn.LayerNorm(size))
        elif norm_type == "batch":
            layers.append(nn.BatchNorm1d(size))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class scVAE(nn.Module):
    """
    A basic single-cell VAE to test things out on.
    """

    def __init__(
        self,
        n_features: int,
        n_batches: int = 1,
        latent_dim: int = 10,
        gene_likelihood: Literal['nb', 'poisson', 'bernoulli', 'normal'] = 'nb',
        hidden_sizes: Sequence[int] = (128,),
        activation: str = 'relu',
        norm_type: Literal["layer", "batch", "none"] = "batch",
        dropout: float = 0.,
        kl_weight: float = 1.,
        log_normalize: bool = False,
        batch_dispersion: bool = True,
        encoder_batch: bool = True,
        eps: float = 1e-6,
        seed: int = 0,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        super(scVAE, self).__init__()

        self.n_features: int = n_features
        self.n_batches: int = n_batches
        self.latent_dim: int = latent_dim
        self.gene_likelihood: Literal['nb', 'poisson', 'bernoulli', 'normal'] = gene_likelihood
        self.kl_weight: float = kl_weight
        self.log_normalize: bool = log_normalize
        self.batch_dispersion: bool = batch_dispersion
        self.encoder_batch: bool = encoder_batch
        self.eps: float = eps
        self.device: torch.device = device

        if activation not in activations:
            logger.warn("Choice of neural network activation not found.")
        
        
        encoder_input_dim = (self.n_features + self.n_batches) if encoder_batch and self.n_batches > 1 else self.n_features
        decoder_input_dim = (self.latent_dim + self.n_batches) if self.n_batches > 1 else self.latent_dim
        self.encoder: nn.Module = get_fully_connected_layers(
            encoder_input_dim,
            self.latent_dim * 2,
            hidden_sizes,
            activation=activation,
            norm_type=norm_type,
            dropout_prob=dropout
        )

        self.decoder: nn.Module = get_fully_connected_layers(
            decoder_input_dim,
            self.n_features,
            hidden_sizes[::-1],
            activation=activation,
            norm_type=norm_type,
            dropout_prob=dropout
        )

        self.softmax = nn.Softmax(dim=-1)

        if self.batch_dispersion and self.gene_likelihood == 'nb' and self.n_batches > 1:
            self.dispersion: nn.Parameter = nn.Parameter(
                torch.rand(self.n_batches, self.n_features), requires_grad=True
            )
        elif self.gene_likelihood == 'nb':
            self.dispersion: nn.Parameter = nn.Parameter(
                torch.rand(self.n_features), requires_grad=True
            )
        
        self.to(self.device)

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor]):
        if self.batch_dispersion and self.n_batches > 1:
            assert batch is not None

        library_size = x.sum(keepdim=True, dim=-1)

        if self.log_normalize:
            x_in = x / library_size
            x_in = torch.log1p(x_in)
        else:
            x_in = x
        
        if self.n_batches > 1:
            oh = F.one_hot(batch, num_classes=self.n_batches)
            if self.encoder_batch:
                x_in = torch.cat([x_in, oh], dim=1)

        encode = self.encoder(x_in)
        z_mu, z_log_var = encode[:, :self.latent_dim], encode[:, self.latent_dim:]
        
        q = Normal(z_mu, F.softplus(z_log_var) + self.eps)
        if self.training:
            z = q.rsample()
        else:
            z = z_mu

        z_in = torch.cat([z, oh], dim=1) if self.n_batches > 1 else z
        recon_mu = self.decoder(z_in)

        if self.gene_likelihood == 'bernoulli':
            recon_mu = F.sigmoid(recon_mu)
        else:
            recon_mu = self.softmax(recon_mu) * library_size

        return z, z_mu, z_log_var, recon_mu
    
    def loss_function(
        self,
        x: torch.Tensor, # unnormalized counts
        z: torch.Tensor, # latent representation
        z_mu: torch.Tensor, # latent means
        z_log_var: torch.Tensor, # latent variances
        recon_mu: torch.Tensor, # decoded values
        batch: Optional[torch.Tensor], # batch indices
    ):
        # import pdb; pdb.set_trace()
        p_mu, p_var = torch.zeros(z_mu.size()[0], self.latent_dim).to(self.device), \
            torch.ones(z_mu.size()[0], self.latent_dim).to(self.device)
        p = Normal(p_mu, p_var)
        q = Normal(z_mu, F.softplus(z_log_var) + self.eps)
        kl_loss = torch.mean(torch.distributions.kl_divergence(p, q), dim=-1).mean()

        if self.gene_likelihood == 'nb':
            if self.batch_dispersion and self.n_batches > 1:
                dispersion = F.softplus(self.dispersion[batch]) + self.eps
            else:
                dispersion = F.softplus(self.dispersion) + self.eps
            recon_loss = -log_nb_positive(x, recon_mu, dispersion).mean() / self.n_features
        elif self.gene_likelihood == 'normal':
            recon_loss = F.mse_loss(recon_mu, x).mean()
        elif self.gene_likelihood == 'poisson':
            recon_loss = F.poisson_nll_loss(recon_mu, x, log_input=False)
        elif self.gene_likelihood == 'bernoulli':
            recon_loss = F.binary_cross_entropy(
                recon_mu, x, reduction='none' # Assume that x is already binarized
            ).sum(-1).mean()

        return self.kl_weight * kl_loss + recon_loss, kl_loss, recon_loss
