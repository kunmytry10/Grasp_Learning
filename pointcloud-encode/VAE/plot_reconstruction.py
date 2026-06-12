"""
Reconstruction quality visualization for VAE.

Shows 10 test samples: original point cloud (blue) vs VAE reconstruction (red),
with class name and per-sample Chamfer Distance.

Usage:
    cd pointcloud-encode/VAE
    python plot_reconstruction.py \
        --checkpoint log/vae/2026-06-11_17-57/checkpoints/best_model.pth \
        --use_normals --gpu 0
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import importlib

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PN2_DIR = os.path.join(THIS_DIR, '..', '..', '..', 'Pointnet_Pointnet2_pytorch')
sys.path.insert(0, PN2_DIR)
sys.path.insert(0, os.path.join(PN2_DIR, 'models'))
sys.path.insert(0, os.path.join(PN2_DIR, 'data_utils'))

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from pointnet2_vae import chamfer_distance

OUT_DIR = os.path.join(THIS_DIR, 'log', 'vae', 'analysis')
VAE_MODEL_NAME = 'pointnet2_vae'

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser('VAE Reconstruction Visualization')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--num_category', type=int, default=40)
    parser.add_argument('--use_normals', action='store_true', default=False)
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--use_uniform_sample', action='store_true',
                        default=False)
    parser.add_argument('--process_data', action='store_true', default=False)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Collect samples (one per class for diversity)
# ---------------------------------------------------------------------------

def collect_samples(dataloader, num_samples, device, model, class_names):
    """Collect samples, prioritizing distinct classes where possible."""
    model.eval()
    seen_classes = set()
    samples = []  # list of (points_batch, label_idx)

    with torch.no_grad():
        for points, labels in dataloader:
            for i in range(points.size(0)):
                label = labels[i].item()
                if label not in seen_classes:
                    seen_classes.add(label)
                    samples.append((points[i:i+1].clone(), label))
                    if len(samples) >= num_samples:
                        break
            if len(samples) >= num_samples:
                break

    # If we got fewer than num_samples, pad with random ones
    if len(samples) < num_samples:
        print(f'Warning: Only found {len(samples)} distinct classes, '
              f'need {num_samples}')
        for points, labels in dataloader:
            for i in range(points.size(0)):
                samples.append((points[i:i+1].clone(), labels[i].item()))
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

    return samples[:num_samples]


# ---------------------------------------------------------------------------
# 3D Plot
# ---------------------------------------------------------------------------

def plot_reconstructions(samples, model, device, class_names, save_path):
    """
    For each sample, run through VAE and plot original vs reconstruction.
    2 rows × 5 cols grid.
    """
    model.eval()
    n = len(samples)
    ncols = 5
    nrows = 2

    fig = plt.figure(figsize=(20, 9))

    # Normalization range for consistent axis limits across subplots
    all_points = []

    results = []
    with torch.no_grad():
        for points_batch, label in samples:
            points_batch = points_batch.to(device)
            points_6ch = points_batch.transpose(2, 1)  # [1, C, N]
            result = model(points_6ch)
            recon = result['recon'].cpu().numpy()[0]    # [1024, 3]
            original = result['xyz'].cpu().numpy()[0]   # [1024, 3]

            # Per-sample CD
            cd_val = chamfer_distance(
                torch.tensor(recon).unsqueeze(0),
                torch.tensor(original).unsqueeze(0)
            ).item()

            results.append((original, recon, cd_val, label))
            all_points.append(original)
            all_points.append(recon)

    # Global scale for consistent view
    all_pts = np.concatenate(all_points, axis=0)
    global_max = np.abs(all_pts).max() * 1.05

    for idx, (original, recon, cd_val, label) in enumerate(results):
        row, col = idx // ncols, idx % ncols
        # Original (left sub-plot in pair)
        ax_orig = fig.add_subplot(nrows, ncols * 2, row * ncols * 2 + col * 2 + 1,
                                  projection='3d')
        ax_orig.scatter(original[:, 0], original[:, 1], original[:, 2],
                        c='steelblue', s=1.5, alpha=0.8, edgecolors='none')
        ax_orig.set_title(f'{class_names[label].replace("_", " ")}\n(Original)',
                          fontsize=8, fontweight='bold')
        ax_orig.set_xlim(-global_max, global_max)
        ax_orig.set_ylim(-global_max, global_max)
        ax_orig.set_zlim(-global_max, global_max)
        ax_orig.set_xticklabels([])
        ax_orig.set_yticklabels([])
        ax_orig.set_zticklabels([])
        ax_orig.view_init(elev=20, azim=60)

        # Reconstructed (right sub-plot in pair)
        ax_recon = fig.add_subplot(nrows, ncols * 2, row * ncols * 2 + col * 2 + 2,
                                   projection='3d')
        ax_recon.scatter(recon[:, 0], recon[:, 1], recon[:, 2],
                         c='firebrick', s=1.5, alpha=0.8, edgecolors='none')
        ax_recon.set_title(f'{class_names[label].replace("_", " ")}\n(Recon, CD={cd_val:.4f})',
                           fontsize=8, fontweight='bold')
        ax_recon.set_xlim(-global_max, global_max)
        ax_recon.set_ylim(-global_max, global_max)
        ax_recon.set_zlim(-global_max, global_max)
        ax_recon.set_xticklabels([])
        ax_recon.set_yticklabels([])
        ax_recon.set_zticklabels([])
        ax_recon.view_init(elev=20, azim=60)

    fig.suptitle('VAE Point Cloud Reconstruction — Original (Blue) vs Reconstructed (Red)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if (torch.cuda.is_available()
                           and not args.use_cpu) else 'cpu')
    print(f'Device: {device}')
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load data ----
    print('Loading ModelNet40 test set...')
    data_path = os.path.join(PN2_DIR, 'data', 'modelnet40_normal_resampled')
    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split='test',
        process_data=args.process_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    class_names = test_dataset.cat
    print(f'  Classes: {len(class_names)}')

    # ---- Load VAE model ----
    print(f'Loading VAE from: {args.checkpoint}')
    vae_ckpt = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
    vae_args = vae_ckpt.get('args', None)
    vae_lib = importlib.import_module(VAE_MODEL_NAME)
    vae = vae_lib.get_model(
        latent_dim=vae_args.latent_dim if vae_args else 256,
        normal_channel=vae_args.use_normals if vae_args else args.use_normals,
        num_output_points=vae_args.num_output_points if vae_args else 1024,
    ).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    print(f'  Epoch: {vae_ckpt.get("epoch", -1)}, '
          f'Test CD: {vae_ckpt.get("test_cd", -1):.6f}')

    # ---- Collect samples ----
    print(f'Collecting {args.num_samples} samples...')
    samples = collect_samples(test_loader, args.num_samples, device,
                              vae, class_names)
    print(f'  Collected {len(samples)} samples')
    for _, label in samples:
        print(f'    {class_names[label]}')

    # ---- Plot ----
    print('Plotting reconstructions...')
    plot_reconstructions(samples, vae, device, class_names,
                         os.path.join(OUT_DIR, 'reconstruction_samples.png'))

    # ---- Summary stats ----
    # Compute average CD on these samples
    cds = []
    with torch.no_grad():
        for points_batch, _ in samples:
            points_6ch = points_batch.to(device).transpose(2, 1)
            result = vae(points_6ch)
            cd = chamfer_distance(result['recon'], result['xyz']).item()
            cds.append(cd)
    print(f'\n  Mean CD (10 samples): {np.mean(cds):.6f}')
    print(f'  Min  CD:              {np.min(cds):.6f}')
    print(f'  Max  CD:              {np.max(cds):.6f}')
    print('Done.')


if __name__ == '__main__':
    main()
