from typing import Optional

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import spmatrix
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from vae import scVAE


class Trainer:
    """
    Simple trainer class for the basic VAE
    """
    def __init__(
        self,
        model: scVAE,
        adata: ad.AnnData,
        batch_col: Optional[str] = None,  # If None, assume there are no batches
        use_highly_variable: bool = False,
        layer: Optional[str] = None,  # If None, use X
        batch_size: int = 2048,
        seed: int = 0
    ):
        self.adata: ad.AnnData = adata
        self.model: scVAE = model
        self.batch_col: Optional[str] = batch_col
        self.use_highly_variable: bool = use_highly_variable
        self.layer: Optional[str] = layer
        self.batch_size: int = batch_size

        if self.use_highly_variable and 'highly_variable' not in adata.var.columns:
            raise ValueError('"highly_variable" was not found in adata.var')

        X = self._process_adata(adata)
        
        if self.batch_col is not None:
            B = torch.Tensor(adata.obs[batch_col].astype('category').cat.codes)
            self.dataset: Dataset = TensorDataset(X, B)
        else:
            self.dataset: Dataset = TensorDataset(X)

        self.dataloader: DataLoader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
    
    def _process_adata(self, adata: ad.AnnData):
        if self.layer is not None:
            X = adata.layers[self.layer]
        else:
            X = adata.X
        
        if isinstance(X, spmatrix):
            X = X.toarray()

        if self.use_highly_variable:
            X = X[:, adata.var.highly_variable]

        X = torch.Tensor(X)
        return X

    def train(
        self,
        n_epochs: int,
        optimizer: optim.Optimizer
    ):
        pbar = tqdm(range(1, n_epochs + 1))
        for epoch in pbar:
            total_loss, total_kl_loss, total_recon_loss = self._train_one_epoch(optimizer)

            pbar.set_description(
                f'Loss: {total_loss:.3f}, KL: {total_kl_loss:.3f}, Recon: {total_recon_loss:.3f}'
            )

    def _train_one_epoch(self, optimizer: optim.Optimizer):
        self.model.train()

        total_loss = total_kl_loss = total_recon_loss = 0
        n_batches = 0

        for i, data in enumerate(self.dataloader):
            if len(data) == 2:
                x, batch = data
                batch = batch.to(self.model.device)
            else:
                x = data[0]
                batch = None
            
            x = x.to(self.model.device)

            z, z_mu, z_log_var, recon_mu = self.model(x, batch)
            loss, kl_loss, recon_loss = self.model.loss_function(
                x, z, z_mu, z_log_var, recon_mu, batch
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 15)
            optimizer.step()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_recon_loss += recon_loss.item()
            n_batches += 1

        return total_loss / n_batches, \
            total_kl_loss / n_batches, \
            total_recon_loss / n_batches

    def get_latent_representation(
        self,
        adata: Optional[ad.AnnData] = None
    ):
        """
        Get the latent representation.
        """
        self.model.eval()

        if adata is None:
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )
        else:
            X = self._process_adata(adata)

            if self.batch_col is not None:
                B = torch.Tensor(adata.obs[self.batch_col].astype('category').cat.codes)
                dataset = TensorDataset(X, B)
            else:
                dataset = TensorDataset(X)

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False
            )

        embs = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if len(data) == 2:
                    x, batch = data
                    batch = batch.to(self.model.device)
                else:
                    x = data[0]
                    batch = None
                
                x = x.to(self.model.device)

                z, _, _, _ = self.model(x, batch)
                embs.append(z.cpu().numpy())
        
        return np.concatenate(embs)
