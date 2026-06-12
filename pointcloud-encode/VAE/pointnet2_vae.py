"""PointNet++ VAE for point cloud feature extraction.

Encoder: PointNet++ SA modules (SSG) -> global feature -> mu/logvar
Decoder: FoldingNet-style 2-fold MLP with fixed 2D grid
Loss: Chamfer Distance + beta * KL divergence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_utils import PointNetSetAbstraction


# ============================================================
# Chamfer Distance
# ============================================================

def chamfer_distance(pred, target):
    """
    Chamfer Distance between two point clouds.

    CD(S1, S2) = mean_{x in S1} min_{y in S2} ||x-y||^2
                + mean_{y in S2} min_{x in S1} ||x-y||^2

    Args:
        pred:   [B, M, 3] predicted point cloud
        target: [B, N, 3] target point cloud
    Returns:
        scalar loss averaged over batch
    """
    dist = torch.cdist(pred, target)                     # [B, M, N]
    dist_forward = torch.min(dist, dim=2)[0]             # [B, M]
    dist_backward = torch.min(dist, dim=1)[0]            # [B, N]
    return torch.mean(dist_forward) + torch.mean(dist_backward)


# ============================================================
# VAE Model
# ============================================================

class get_model(nn.Module):
    def __init__(self, latent_dim=256, normal_channel=True, num_output_points=1024):
        super(get_model, self).__init__()
        self.latent_dim = latent_dim
        self.normal_channel = normal_channel
        self.num_output_points = num_output_points

        in_channel = 6 if normal_channel else 3

        # ---- Encoder: PointNet++ SA modules (SSG) ----
        # SA1: 1024 -> 512 points, radius=0.2, 32 neighbors
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        # SA2: 512 -> 128 points, radius=0.4, 64 neighbors
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # SA3: 128 -> 1 point, global pooling
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # ---- Latent mapping ----
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # ---- Decoder: FoldingNet 2-fold MLP ----
        # Fold 1: (latent_dim + 2) -> 512 -> 256 -> 3
        self.fold1 = nn.Sequential(
            nn.Linear(latent_dim + 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        # Fold 2: (latent_dim + 3) -> 512 -> 256 -> 3
        self.fold2 = nn.Sequential(
            nn.Linear(latent_dim + 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        # ---- Fixed 2D grid (not trainable) ----
        grid = self._build_grid(num_output_points)       # [1, M, 2]
        self.register_buffer('grid', grid)

    def _build_grid(self, npoints):
        """Build a sqrt(npoints) x sqrt(npoints) 2D grid in [-1, 1]."""
        side = int(np.sqrt(npoints))
        if side * side != npoints:
            raise ValueError(
                f"num_output_points ({npoints}) must be a perfect square, "
                f"e.g. 1024 (32x32), 2025 (45x45)")
        x = torch.linspace(-1, 1, side)
        y = torch.linspace(-1, 1, side)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=-1)     # [side, side, 2]
        return grid.view(1, -1, 2)                       # [1, M, 2]

    # ---- Encoder ----

    def encode(self, xyz):
        """
        Args:
            xyz: [B, C, N]  where C=6 (xyz+normals) or C=3 (xyz only)
        Returns:
            global_feat: [B, 1024]
        """
        B = xyz.shape[0]
        if self.normal_channel:
            norm = xyz[:, 3:, :]        # [B, 3, N]
            xyz = xyz[:, :3, :]         # [B, 3, N]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points.view(B, -1)    # [B, 1024]

    # ---- Reparameterization ----

    def reparameterize(self, mu, logvar):
        """
        Sample z from N(mu, sigma^2) using the reparameterization trick.
        Training: z = mu + sigma * eps,   eps ~ N(0,1)
        Eval:     z = mu (deterministic)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    # ---- Decoder ----

    def decode(self, z):
        """
        Args:
            z: [B, latent_dim]
        Returns:
            recon: [B, M, 3] reconstructed point cloud
        """
        B = z.shape[0]
        M = self.num_output_points

        # Fold 1: z + 2D grid -> coarse shape
        z_exp = z.unsqueeze(1).expand(-1, M, -1)          # [B, M, latent_dim]
        grid = self.grid.expand(B, -1, -1)                 # [B, M, 2]
        fold1_in = torch.cat([z_exp, grid], dim=-1)        # [B, M, latent_dim+2]
        fold1_in = fold1_in.reshape(B * M, -1)             # [B*M, latent_dim+2]
        fold1_out = self.fold1(fold1_in)                   # [B*M, 3]
        fold1_out = fold1_out.view(B, M, 3)                # [B, M, 3]

        # Fold 2: z + coarse shape -> refined shape
        z_exp2 = z.unsqueeze(1).expand(-1, M, -1)          # [B, M, latent_dim]
        fold2_in = torch.cat([z_exp2, fold1_out], dim=-1)  # [B, M, latent_dim+3]
        fold2_in = fold2_in.reshape(B * M, -1)             # [B*M, latent_dim+3]
        fold2_out = self.fold2(fold2_in)                   # [B*M, 3]
        return fold2_out.view(B, M, 3)                     # [B, M, 3]

    # ---- Forward ----

    def forward(self, xyz):
        """
        Args:
            xyz: [B, C, N]  point cloud (C=6 with normals, C=3 without)
        Returns:
            dict with keys: recon [B,M,3], xyz [B,N,3], mu [B,latent_dim],
                            logvar [B,latent_dim]
        """
        # Save original xyz as reconstruction target (no normals)
        xyz_target = xyz[:, :3, :].transpose(1, 2).contiguous()  # [B, N, 3]

        global_feat = self.encode(xyz)              # [B, 1024]
        mu = self.fc_mu(global_feat)                 # [B, latent_dim]
        logvar = self.fc_logvar(global_feat)          # [B, latent_dim]
        # Clamp to prevent numerical explosion (exp(logvar/2) would overflow)
        # [-4, 4] -> sigma range [exp(-2)≈0.14, exp(2)≈7.4], safe for sampling
        logvar = torch.clamp(logvar, min=-4, max=4)
        z = self.reparameterize(mu, logvar)          # [B, latent_dim]
        recon = self.decode(z)                       # [B, M, 3]

        return {
            'recon': recon,
            'xyz': xyz_target,
            'mu': mu,
            'logvar': logvar,
        }


# ============================================================
# Loss
# ============================================================

class get_loss(nn.Module):
    """
    VAE loss: Chamfer Distance (reconstruction) + beta * KL divergence.
    """
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred_dict, beta=1.0):
        """
        Args:
            pred_dict: output of get_model.forward()
            beta: KL annealing weight (0.0 = AE, 1.0 = full VAE)
        Returns:
            dict with 'total', 'recon', 'kl'
        """
        recon = pred_dict['recon']
        xyz_target = pred_dict['xyz']
        mu = pred_dict['mu']
        logvar = pred_dict['logvar']

        loss_recon = chamfer_distance(recon, xyz_target)

        # KL divergence, normalized by latent_dim for unit-scale KL (~O(1))
        latent_dim = mu.shape[1]
        loss_kl = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        ) / latent_dim

        loss_total = loss_recon + beta * loss_kl

        return {
            'total': loss_total,
            'recon': loss_recon,
            'kl': loss_kl,
        }
