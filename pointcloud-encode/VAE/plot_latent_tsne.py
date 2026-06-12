"""
t-SNE visualization: VAE latent space vs PointNet++ latent space.

Two figures:
  1. tsne_comparison.png  — 2-panel t-SNE (PointNet++ left, VAE right)
  2. similarity_comparison.png — 2-panel 40×40 cosine similarity matrix

Usage:
    cd pointcloud-encode/VAE
    python plot_latent_tsne.py \
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
import seaborn as sns
import argparse
import importlib

from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PN2_DIR = os.path.join(THIS_DIR, '..', '..', '..', 'Pointnet_Pointnet2_pytorch')
sys.path.insert(0, PN2_DIR)
sys.path.insert(0, os.path.join(PN2_DIR, 'models'))
sys.path.insert(0, os.path.join(PN2_DIR, 'data_utils'))

from data_utils.ModelNetDataLoader import ModelNetDataLoader

OUT_DIR = os.path.join(THIS_DIR, 'log', 'vae', 'analysis')
VAE_MODEL_NAME = 'pointnet2_vae'

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser('VAE vs PointNet++ Latent Space t-SNE')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to VAE best_model.pth')
    parser.add_argument('--pn2_checkpoint', type=str,
                        default=os.path.join(
                            PN2_DIR, 'log', 'classification',
                            'pointnet2_msg_normals', 'checkpoints',
                            'best_model.pth'),
                        help='Path to PointNet++ classification checkpoint')
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
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_vae_features(model, dataloader, device):
    """Extract VAE mu vectors (256-dim) from test set."""
    model.eval()
    all_mu, all_labels = [], []
    with torch.no_grad():
        for points, target in tqdm(dataloader, desc='VAE features'):
            points = points.transpose(2, 1).to(device)
            global_feat = model.encode(points)      # [B, 1024]
            mu = model.fc_mu(global_feat)            # [B, 256]
            all_mu.append(mu.cpu().numpy())
            all_labels.append(target.numpy())
    return (np.concatenate(all_mu, axis=0),
            np.concatenate(all_labels, axis=0))


def extract_pn2_features(model, dataloader, device):
    """Extract PointNet++ l3_points (1024-dim) from test set."""
    model.eval()
    all_feat, all_labels = [], []
    with torch.no_grad():
        for points, target in tqdm(dataloader, desc='PointNet++ features'):
            points = points.transpose(2, 1).to(device)
            _, l3_points = model(points)             # (B, 40), (B, 1024, 1)
            feat = l3_points.squeeze(-1)              # [B, 1024]
            all_feat.append(feat.cpu().numpy())
            all_labels.append(target.numpy())
    return (np.concatenate(all_feat, axis=0),
            np.concatenate(all_labels, axis=0))


# ---------------------------------------------------------------------------
# t-SNE Plot
# ---------------------------------------------------------------------------

def plot_tsne_comparison(feat_pn2, labels_pn2, feat_vae, labels_vae,
                         class_names, save_path):
    """2-panel t-SNE: PointNet++ (left) vs VAE (right)."""
    print('Running t-SNE on PointNet++ features (1024-dim)...')
    tsne_pn2 = TSNE(n_components=2, perplexity=30, random_state=42,
                    max_iter=1000, verbose=1).fit_transform(feat_pn2)

    print('Running t-SNE on VAE features (256-dim)...')
    tsne_vae = TSNE(n_components=2, perplexity=30, random_state=42,
                    max_iter=1000, verbose=1).fit_transform(feat_vae)

    colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    short_names = [n.replace('_', ' ') for n in class_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 11))

    for ax, tsne_res, labels, title, dim in [
        (ax1, tsne_pn2, labels_pn2,
         'PointNet++ Classifier\n(1024-dim, supervised, 93% acc)', 1024),
        (ax2, tsne_vae, labels_vae,
         'VAE Encoder\n(256-dim μ, self-supervised, CD=0.125)', 256)
    ]:
        for c in range(40):
            mask = labels == c
            ax.scatter(tsne_res[mask, 0], tsne_res[mask, 1],
                       c=[colors[c]], s=6, alpha=0.5,
                       label=short_names[c])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')
        ax.grid(alpha=0.2)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
                  fontsize=6, ncol=2, frameon=True,
                  markerscale=2, borderpad=0.5, labelspacing=0.3)

    fig.suptitle('Latent Space t-SNE: PointNet++ vs VAE (ModelNet40 Test, 2468 samples)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')


# ---------------------------------------------------------------------------
# Similarity Matrix Plot
# ---------------------------------------------------------------------------

def compute_similarity_matrix(features, labels, class_names):
    """Class-mean cosine similarity: 40×40 matrix."""
    n_classes = len(class_names)
    centroids = np.zeros((n_classes, features.shape[1]))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = features[mask].mean(axis=0)
    return cosine_similarity(centroids)


def plot_similarity_comparison(sim_pn2, sim_vae, class_names, save_path):
    """2-panel heatmap: PointNet++ vs VAE."""
    short_names = [n.replace('_', ' ') for n in class_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))

    for ax, sim_mat, title in [
        (ax1, sim_pn2,
         'PointNet++ Classifier\n(supervised, cross-entropy training)'),
        (ax2, sim_vae,
         'VAE Encoder\n(self-supervised, reconstruction + KL)')
    ]:
        sns.heatmap(sim_mat, ax=ax, cmap='RdBu_r', center=0.5,
                    vmin=0, vmax=1, square=True,
                    xticklabels=short_names, yticklabels=short_names,
                    cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Class')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                           ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.suptitle('Class Similarity Matrix: PointNet++ vs VAE',
                 fontsize=16, fontweight='bold', y=1.01)
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
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    class_names = test_dataset.cat
    print(f'  Samples: {len(test_dataset)}, Classes: {len(class_names)}')

    # ---- Load VAE model ----
    print(f'\nLoading VAE from: {args.checkpoint}')
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

    # ---- Load PointNet++ Classifier ----
    print(f'\nLoading PointNet++ from: {args.pn2_checkpoint}')
    pn2_lib = importlib.import_module('pointnet2_cls_msg')
    pn2 = pn2_lib.get_model(
        args.num_category, normal_channel=args.use_normals).to(device)
    pn2_ckpt = torch.load(args.pn2_checkpoint, map_location=device,
                          weights_only=False)
    pn2.load_state_dict(pn2_ckpt['model_state_dict'])
    print(f'  Epoch: {pn2_ckpt.get("epoch", -1)}, '
          f'Acc: {pn2_ckpt.get("acc", -1):.2f}%')

    # ---- Extract features ----
    print('\n=== Extracting Features ===')
    feat_pn2, labels_pn2 = extract_pn2_features(pn2, test_loader, device)
    feat_vae, labels_vae = extract_vae_features(vae, test_loader, device)
    print(f'  PointNet++: {feat_pn2.shape}')
    print(f'  VAE:         {feat_vae.shape}')

    # ---- t-SNE ----
    print('\n=== t-SNE ===')
    plot_tsne_comparison(
        feat_pn2, labels_pn2, feat_vae, labels_vae, class_names,
        os.path.join(OUT_DIR, 'tsne_comparison.png'))

    # ---- Similarity Matrix ----
    print('\n=== Similarity Matrix ===')
    sim_pn2 = compute_similarity_matrix(feat_pn2, labels_pn2, class_names)
    sim_vae = compute_similarity_matrix(feat_vae, labels_vae, class_names)
    plot_similarity_comparison(
        sim_pn2, sim_vae, class_names,
        os.path.join(OUT_DIR, 'similarity_comparison.png'))

    # ---- Quick stats ----
    print('\n=== Similarity Stats ===')
    # Off-diagonal mean (how much classes bleed into each other)
    mask_off = ~np.eye(40, dtype=bool)
    print(f'  PointNet++ off-diag mean sim: {sim_pn2[mask_off].mean():.4f}')
    print(f'  VAE         off-diag mean sim: {sim_vae[mask_off].mean():.4f}')
    # Top confused pairs
    sim_pn2_off = sim_pn2.copy()
    np.fill_diagonal(sim_pn2_off, 0)
    sim_vae_off = sim_vae.copy()
    np.fill_diagonal(sim_vae_off, 0)
    top_pn2 = np.unravel_index(np.argmax(sim_pn2_off), sim_pn2.shape)
    top_vae = np.unravel_index(np.argmax(sim_vae_off), sim_vae.shape)
    print(f'  PointNet++ most similar: '
          f'{class_names[top_pn2[0]]} ↔ {class_names[top_pn2[1]]} '
          f'({sim_pn2[top_pn2]:.4f})')
    print(f'  VAE         most similar: '
          f'{class_names[top_vae[0]]} ↔ {class_names[top_vae[1]]} '
          f'({sim_vae[top_vae]:.4f})')

    print('\nDone! 2 figures saved.')


if __name__ == '__main__':
    main()
