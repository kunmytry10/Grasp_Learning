"""
Latent space arithmetic visualization for VAE.
PCA-based interactive latent space browsing.

Outputs (saved to analysis/pca_arithmetic/):
  - pca_explained_variance.png    PCA explained variance bar chart
  - pc1_sweep.gif, pc2_sweep.gif, pc3_sweep.gif   Single-direction sweeps
  - grid_2d_pc1_pc2.png          5×5 grid static image
  - grid_2d_rotation.gif         Grid rotation animation
  - multi_step_path.gif          A→B→C polyline path animation

Usage:
    cd ~/code/Grasp_Learning/pointcloud-encode/VAE
    python plot_latent_arithmetic.py \
        --checkpoint log/vae/2026-06-11_17-57/checkpoints/best_model.pth
"""

import os, sys, argparse, io
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from sklearn.decomposition import PCA

# ---- Path setup ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PN2_DIR = os.path.join(THIS_DIR, '..', '..', '..', 'Pointnet_Pointnet2_pytorch')
sys.path.insert(0, PN2_DIR)
sys.path.insert(0, os.path.join(PN2_DIR, 'models'))
sys.path.insert(0, os.path.join(PN2_DIR, 'data_utils'))

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from pointnet2_vae import get_model

OUT_DIR = os.path.join(THIS_DIR, 'log', 'vae', 'analysis', 'pca_arithmetic')

# Anchor classes for visualization: pick recognizable, diverse shapes
ANCHOR_CLASSES = ['chair', 'table', 'airplane', 'lamp', 'sofa']
MULTI_STEP_CLASSES = ['chair', 'table', 'airplane']  # for A→B→C path


def parse_args():
    p = argparse.ArgumentParser('VAE Latent Arithmetic — PCA Latent Browsing')
    p.add_argument('--checkpoint', type=str,
                   default='log/vae/2026-06-11_17-57/checkpoints/best_model.pth')
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--num_point', type=int, default=1024)
    p.add_argument('--num_category', type=int, default=40)
    p.add_argument('--use_normals', action='store_true', default=False)
    p.add_argument('--use_cpu', action='store_true', default=False)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--use_uniform_sample', action='store_true', default=False)
    p.add_argument('--process_data', action='store_true', default=False)
    p.add_argument('--n_sweep_frames', type=int, default=15,
                    help='Frames per sweep GIF')
    p.add_argument('--sweep_alpha', type=float, default=3.0,
                    help='Alpha range for sweeps (±)')
    p.add_argument('--grid_size', type=int, default=5,
                    help='Grid size (N×N) for 2D latent map')
    p.add_argument('--grid_alpha', type=float, default=2.0,
                    help='Alpha range for 2D grid (±)')
    p.add_argument('--n_rotation_frames', type=int, default=24,
                    help='Frames for grid rotation GIF')
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  Helper: find a sample from a specific class
# ══════════════════════════════════════════════════════════════════════

def find_sample(dataloader, class_name, class_names, device):
    """Find one sample from class_name. Returns (points [1,C,N], label_idx)."""
    target_label = class_names.index(class_name)
    for points, labels in dataloader:
        for i in range(points.size(0)):
            if labels[i].item() == target_label:
                pt = points[i:i+1].clone().transpose(2, 1)  # [1,6,N]
                return pt.to(device), target_label
    raise RuntimeError(f'Could not find sample for class: {class_name}')


def find_centroid_sample(dataloader, class_name, class_names, device, model):
    """
    Find the sample closest to the class mean in latent space.
    Returns (points [1,C,N], mu [1,256], label_idx).
    """
    target_label = class_names.index(class_name)
    all_pts, all_mus = [], []
    for points, labels in dataloader:
        for i in range(points.size(0)):
            if labels[i].item() == target_label:
                pt = points[i:i+1].clone().transpose(2, 1).to(device)  # [1,6,N]
                with torch.no_grad():
                    mu = model.fc_mu(model.encode(pt))
                all_pts.append(pt.cpu())
                all_mus.append(mu.cpu())
    if not all_pts:
        raise RuntimeError(f'Could not find samples for class: {class_name}')
    all_mus = torch.cat(all_mus, dim=0)  # [K, 256]
    centroid = all_mus.mean(dim=0, keepdim=True)  # [1, 256]
    dists = torch.norm(all_mus - centroid, dim=1)
    closest_idx = dists.argmin().item()
    return all_pts[closest_idx].to(device), all_mus[closest_idx:closest_idx+1].to(device), target_label


# ══════════════════════════════════════════════════════════════════════
#  Helper: encode entire dataset to mu vectors
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_dataset(dataloader, model, device):
    """Encode all samples to mu vectors. Returns all_mu [N, 256], all_labels [N]."""
    model.eval()
    all_mu, all_labels, all_pts = [], [], []
    for points, labels in dataloader:
        points = points.transpose(2, 1).to(device)  # [B, 6, N]
        mu = model.fc_mu(model.encode(points))        # [B, 256]
        all_mu.append(mu.cpu())
        all_labels.append(labels)
    all_mu = torch.cat(all_mu, dim=0).numpy()          # [N, 256]
    all_labels = torch.cat(all_labels, dim=0).numpy()   # [N]
    print(f'  Encoded {len(all_mu)} samples → shape {all_mu.shape}')
    return all_mu, all_labels


# ══════════════════════════════════════════════════════════════════════
#  Helper: render a single point cloud to a numpy array
# ══════════════════════════════════════════════════════════════════════

def render_point_cloud(pts_xyz, title='', color='#00bcd4',
                       elev=25, azim=60, figsize=(4, 4),
                       facecolor='#1a1a2e', lim=None, s=3, alpha=0.9,
                       show_title=True, title_color='#e0e0e0'):
    """Render a single point cloud to an image array."""
    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           subplot_kw={'projection': '3d'},
                           facecolor=facecolor)
    ax.set_facecolor(facecolor)
    ax.scatter(pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2],
               c=[color], s=s, alpha=alpha, edgecolors='none')

    if lim is None:
        lim = np.abs(pts_xyz).max() * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333355')
    ax.yaxis.pane.set_edgecolor('#333355')
    ax.zaxis.pane.set_edgecolor('#333355')
    ax.grid(False)

    if show_title and title:
        ax.set_title(title, fontsize=10, fontweight='bold', color=title_color)

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=facecolor)
    buf.seek(0)
    img = imageio.v2.imread(buf)
    plt.close(fig)
    buf.close()
    return img


# ══════════════════════════════════════════════════════════════════════
#  Figure 1: PCA explained variance bar chart
# ══════════════════════════════════════════════════════════════════════

def plot_explained_variance(pca, save_path):
    """Plot top-20 principal components explained variance ratio."""
    n = min(20, len(pca.explained_variance_ratio_))
    ratios = pca.explained_variance_ratio_[:n] * 100
    cumsum = np.cumsum(ratios)

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    bars = ax1.bar(range(1, n+1), ratios, color='#5c6bc0', edgecolor='#3f51b5', linewidth=0.5)
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio (%)', fontsize=12, fontweight='bold',
                   color='#5c6bc0')
    ax1.tick_params(axis='y', labelcolor='#5c6bc0')
    ax1.set_xticks(range(1, n+1))
    ax1.set_xticklabels([f'PC{i}' for i in range(1, n+1)], rotation=45)

    # Annotate bars with percentage
    for bar, val in zip(bars, ratios):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Cumulative line
    ax2 = ax1.twinx()
    ax2.plot(range(1, n+1), cumsum, 'o-', color='#ef5350', linewidth=2, markersize=6)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12, fontweight='bold',
                   color='#ef5350')
    ax2.tick_params(axis='y', labelcolor='#ef5350')
    ax2.set_ylim(0, 105)

    ax1.set_title('PCA on VAE Latent Space (ModelNet40 Test Set)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {os.path.basename(save_path)}')


# ══════════════════════════════════════════════════════════════════════
#  Figure 2-4: Single-direction sweep GIFs (PC1, PC2, PC3)
# ══════════════════════════════════════════════════════════════════════

def make_sweep_gif(model, mu_anchor, pca, pc_idx, sigma_k, anchor_name,
                   save_path, n_frames=15, alpha_range=3.0, device='cuda'):
    """
    Sweep along one principal direction and generate GIF.

    Args:
        mu_anchor: [1, 256] anchor latent vector
        pca: fitted sklearn PCA object
        pc_idx: 0-based principal component index (0=PC1, 1=PC2, ...)
        sigma_k: standard deviation along this PC (for step scaling)
    """
    direction = torch.tensor(pca.components_[pc_idx], dtype=torch.float32,
                             device=device).unsqueeze(0)  # [1, 256]
    alphas = np.linspace(-alpha_range, alpha_range, n_frames)

    frames = []
    with torch.no_grad():
        for alpha in alphas:
            z = mu_anchor + alpha * sigma_k * direction  # [1, 256]
            recon = model.decode(z)                       # [1, 1024, 3]
            pts = recon.squeeze(0).cpu().numpy()          # [1024, 3]
            t_norm = (alpha + alpha_range) / (2 * alpha_range)  # 0..1
            color = plt.cm.coolwarm(t_norm)
            img = render_point_cloud(
                pts, title=f'PC{pc_idx+1}  α={alpha:+.1f}', color=color,
                elev=25, azim=60, figsize=(4, 4), s=2)
            frames.append(img)

    _save_gif(save_path, frames, duration=250)
    print(f'  Saved: {os.path.basename(save_path)} ({len(frames)} frames)')


# ══════════════════════════════════════════════════════════════════════
#  Utility: save frames as GIF with consistent sizing
# ══════════════════════════════════════════════════════════════════════

def _save_gif(save_path, frames, duration=250):
    """Resize frames to consistent shape, then save as GIF."""
    import PIL.Image
    target_h, target_w = frames[0].shape[:2]
    resized = []
    for img in frames:
        if img.shape[:2] != (target_h, target_w):
            pil = PIL.Image.fromarray(img)
            pil = pil.resize((target_w, target_h), PIL.Image.LANCZOS)
            img = np.array(pil)
        resized.append(img)
    imageio.mimsave(save_path, resized, duration=duration, loop=0)


# ══════════════════════════════════════════════════════════════════════
#  Figure 5: 2D grid static image (PC1 × PC2)
# ══════════════════════════════════════════════════════════════════════

def make_grid_2d(model, mu_anchor, pca, sigmas, anchor_name,
                 save_path, grid_size=5, alpha_range=2.0, device='cuda'):
    """Generate a grid_size × grid_size static image exploring PC1 × PC2."""
    d1 = torch.tensor(pca.components_[0], dtype=torch.float32,
                      device=device).unsqueeze(0)  # [1, 256]
    d2 = torch.tensor(pca.components_[1], dtype=torch.float32,
                      device=device).unsqueeze(0)  # [1, 256]
    s1, s2 = sigmas[0], sigmas[1]  # std devs

    fig = plt.figure(figsize=(grid_size * 4, grid_size * 4),
                     facecolor='#1a1a2e')
    alphas = np.linspace(-alpha_range, alpha_range, grid_size)

    all_pts_global = []  # For consistent axis limits
    idx = 0

    with torch.no_grad():
        for i, a1 in enumerate(alphas):
            for j, a2 in enumerate(alphas):
                z = mu_anchor + a1 * s1 * d1 + a2 * s2 * d2
                recon = model.decode(z).squeeze(0).cpu().numpy()  # [1024, 3]
                all_pts_global.append(recon)

    global_lim = np.abs(np.concatenate(all_pts_global, axis=0)).max() * 1.1

    idx = 0
    with torch.no_grad():
        for i, a1 in enumerate(alphas):
            for j, a2 in enumerate(alphas):
                z = mu_anchor + a1 * s1 * d1 + a2 * s2 * d2
                recon = model.decode(z).squeeze(0).cpu().numpy()

                # Color: blend based on position in grid
                t1 = (a1 + alpha_range) / (2 * alpha_range)
                t2 = (a2 + alpha_range) / (2 * alpha_range)
                color = plt.cm.coolwarm(0.5 * t1 + 0.5 * t2)

                ax = fig.add_subplot(grid_size, grid_size, idx + 1,
                                     projection='3d')
                ax.set_facecolor('#1a1a2e')
                ax.scatter(recon[:, 0], recon[:, 1], recon[:, 2],
                          c=[color], s=1.5, alpha=0.85, edgecolors='none')
                ax.set_xlim(-global_lim, global_lim)
                ax.set_ylim(-global_lim, global_lim)
                ax.set_zlim(-global_lim, global_lim)
                ax.set_xticklabels([]); ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.view_init(elev=25, azim=60)
                ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('#333355')
                ax.yaxis.pane.set_edgecolor('#333355')
                ax.zaxis.pane.set_edgecolor('#333355')
                ax.grid(False)
                ax.set_title(f'PC1={a1:+.0f}σ  PC2={a2:+.0f}σ',
                           fontsize=8, color='#aaaaaa')
                idx += 1

    fig.suptitle(f'2D Latent Space Map: PC1 × PC2  (anchor: {anchor_name})',
                 fontsize=14, fontweight='bold', color='#e0e0e0', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f'  Saved: {os.path.basename(save_path)} ({grid_size}×{grid_size} grid)')


# ══════════════════════════════════════════════════════════════════════
#  Figure 6: Grid rotation GIF
# ══════════════════════════════════════════════════════════════════════

def make_grid_rotation_gif(model, mu_anchor, pca, sigmas, anchor_name,
                           save_path, grid_size=5, alpha_range=2.0,
                           n_angles=24, device='cuda'):
    """Rotate the 2D grid and save as animation."""
    d1 = torch.tensor(pca.components_[0], dtype=torch.float32,
                      device=device).unsqueeze(0)
    d2 = torch.tensor(pca.components_[1], dtype=torch.float32,
                      device=device).unsqueeze(0)
    s1, s2 = sigmas[0], sigmas[1]
    alphas = np.linspace(-alpha_range, alpha_range, grid_size)

    # Pre-decode all point clouds
    from itertools import product
    all_recons = []
    with torch.no_grad():
        for a1, a2 in product(alphas, alphas):
            z = mu_anchor + a1 * s1 * d1 + a2 * s2 * d2
            recon = model.decode(z).squeeze(0).cpu().numpy()
            all_recons.append((a1, a2, recon))

    global_lim = np.abs(np.concatenate([r[2] for r in all_recons], axis=0)).max() * 1.1

    gif_frames = []
    for azim in np.linspace(0, 360, n_angles, endpoint=False):
        fig = plt.figure(figsize=(grid_size * 3, grid_size * 3),
                        facecolor='#1a1a2e')
        for idx, (a1, a2, recon) in enumerate(all_recons):
            t1 = (a1 + alpha_range) / (2 * alpha_range)
            t2 = (a2 + alpha_range) / (2 * alpha_range)
            color = plt.cm.coolwarm(0.5 * t1 + 0.5 * t2)
            ax = fig.add_subplot(grid_size, grid_size, idx + 1,
                                projection='3d')
            ax.set_facecolor('#1a1a2e')
            ax.scatter(recon[:, 0], recon[:, 1], recon[:, 2],
                      c=[color], s=1.5, alpha=0.85, edgecolors='none')
            ax.set_xlim(-global_lim, global_lim)
            ax.set_ylim(-global_lim, global_lim)
            ax.set_zlim(-global_lim, global_lim)
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.view_init(elev=25, azim=azim)
            ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#333355')
            ax.yaxis.pane.set_edgecolor('#333355')
            ax.zaxis.pane.set_edgecolor('#333355')
            ax.grid(False)
        fig.suptitle(f'2D Latent Map  —  azimuth={azim:.0f}°',
                    fontsize=12, fontweight='bold', color='#e0e0e0', y=1.01)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight',
                   facecolor='#1a1a2e')
        buf.seek(0)
        img = imageio.v2.imread(buf)
        gif_frames.append(img)
        plt.close(fig)
        buf.close()

    _save_gif(save_path, gif_frames, duration=150)
    print(f'  Saved: {os.path.basename(save_path)} ({len(gif_frames)} frames)')


# ══════════════════════════════════════════════════════════════════════
#  Figure 7: Multi-step polyline path (A→B→C)
# ══════════════════════════════════════════════════════════════════════

def make_multistep_gif(model, anchors_mu, class_names_list,
                       save_path, n_steps_per_seg=10, device='cuda'):
    """
    Walk a polyline A→B→C in latent space.
    anchors_mu: list of [1, 256] tensors
    """
    all_zs = []
    segments = len(anchors_mu) - 1
    for seg in range(segments):
        mu_a = anchors_mu[seg]
        mu_b = anchors_mu[seg + 1]
        for i in range(n_steps_per_seg):
            t = i / max(n_steps_per_seg - 1, 1)
            z = (1 - t) * mu_a + t * mu_b
            all_zs.append((seg, t, z))

    frames = []
    # Determine global limits across all decoded clouds
    all_pts = []
    with torch.no_grad():
        for seg, t, z in all_zs:
            recon = model.decode(z).squeeze(0).cpu().numpy()
            all_pts.append(recon)
    global_lim = np.abs(np.concatenate(all_pts, axis=0)).max() * 1.1

    for seg, t, z in all_zs:
        with torch.no_grad():
            recon = model.decode(z).squeeze(0).cpu().numpy()

        total_t = (seg + t) / segments
        color = plt.cm.viridis(total_t)
        seg_name = f'{class_names_list[seg]}→{class_names_list[seg+1]}'

        img = render_point_cloud(
            recon,
            title=f'{seg_name}  t={t:.2f}',
            color=color, elev=25, azim=60, figsize=(4.5, 4.5), s=2)
        frames.append(img)

    # Add transition cards between segments
    final_frames = []
    for i, (seg, t, _) in enumerate(all_zs):
        final_frames.append(frames[i])
        # At segment boundaries, add pause card
        if t > 0.95 and seg < segments - 1:
            card = _make_transition_card(class_names_list[seg],
                                         class_names_list[seg+1])
            final_frames.extend([card] * 3)

    _save_gif(save_path, final_frames, duration=200)
    print(f'  Saved: {os.path.basename(save_path)} ({len(final_frames)} frames)')


def _make_transition_card(from_name, to_name):
    """Create a dark text-only frame for segment transitions."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.text(0.5, 0.55, f'{from_name} → {to_name}',
            fontsize=24, fontweight='bold', ha='center', va='center',
            color='#e0e0e0', transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    img = imageio.v2.imread(buf)
    plt.close(fig)
    buf.close()
    return img


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if (torch.cuda.is_available()
                          and not args.use_cpu) else 'cpu')
    print(f'Device: {device}')
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load checkpoint (need args first to configure dataloader)
    print(f'\n[1/8] Loading VAE checkpoint from: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vae_args = ckpt.get('args', None)
    # Sync args from checkpoint so dataloader matches model's expected input
    if vae_args:
        args.use_normals = vae_args.use_normals
        args.num_point = vae_args.num_point
    print(f'  use_normals={args.use_normals}, num_point={args.num_point}')

    # ── 2. Build model ────────────────────────────────────────────
    print('\n[2/8] Building VAE model...')
    model = get_model(
        latent_dim=vae_args.latent_dim if vae_args else 256,
        normal_channel=vae_args.use_normals if vae_args else args.use_normals,
        num_output_points=vae_args.num_output_points if vae_args else 1024,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Epoch: {ckpt.get("epoch", -1)}, Test CD: {ckpt.get("test_cd", -1):.6f}')

    # ── 3. Load data ──────────────────────────────────────────────
    print('\n[3/8] Loading ModelNet40 test set...')
    data_path = os.path.join(PN2_DIR, 'data', 'modelnet40_normal_resampled')
    test_ds = ModelNetDataLoader(root=data_path, args=args, split='test',
                                  process_data=args.process_data)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    class_names = test_ds.cat
    print(f'  Classes: {len(class_names)}, Samples: {len(test_ds)}')

    # ── 3. Encode dataset → PCA ───────────────────────────────────
    print('\n[4/8] Encoding test set & computing PCA...')
    all_mu, all_labels = encode_dataset(test_loader, model, device)

    n_components = min(20, len(all_mu))
    pca = PCA(n_components=n_components)
    pca.fit(all_mu)
    sigmas = np.sqrt(pca.explained_variance_)  # std dev per PC
    print(f'  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance')
    print(f'  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% variance')
    print(f'  PC3 explains {pca.explained_variance_ratio_[2]*100:.1f}% variance')
    print(f'  Cumsum top-3: {pca.explained_variance_ratio_[:3].sum()*100:.1f}%')
    print(f'  Cumsum top-10: {pca.explained_variance_ratio_[:10].sum()*100:.1f}%')

    # ── 4. Figure 1: Explained variance ───────────────────────────
    print('\n[5/8] Plotting explained variance...')
    plot_explained_variance(pca, os.path.join(OUT_DIR, 'pca_explained_variance.png'))

    # ── 5. Figures 2-4: Single-direction sweeps ───────────────────
    print('\n[6/8] Generating single-direction sweep GIFs...')
    # Use class centroids for stable anchors
    anchor_name = 'chair'
    print(f'  Finding centroid sample: {anchor_name}...')
    anchor_pts, anchor_mu, _ = find_centroid_sample(
        test_loader, anchor_name, class_names, device, model)
    print(f'  Anchor mu norm: {anchor_mu.norm():.3f}')

    for pc_idx in range(3):
        save_path = os.path.join(OUT_DIR, f'pc{pc_idx+1}_sweep.gif')
        make_sweep_gif(model, anchor_mu, pca, pc_idx, sigmas[pc_idx],
                      anchor_name, save_path,
                      n_frames=args.n_sweep_frames,
                      alpha_range=args.sweep_alpha, device=device)

    # ── 6. Figures 5-6: 2D grid ──────────────────────────────────
    print('\n[7/8] Generating 2D grid (PC1×PC2)...')
    save_grid = os.path.join(OUT_DIR, 'grid_2d_pc1_pc2.png')
    make_grid_2d(model, anchor_mu, pca, sigmas, anchor_name,
                save_grid, grid_size=args.grid_size,
                alpha_range=args.grid_alpha, device=device)

    save_rot = os.path.join(OUT_DIR, 'grid_2d_rotation.gif')
    print('  Generating rotation GIF...')
    make_grid_rotation_gif(model, anchor_mu, pca, sigmas, anchor_name,
                          save_rot, grid_size=args.grid_size,
                          alpha_range=args.grid_alpha,
                          n_angles=args.n_rotation_frames, device=device)

    # ── 7. Figure 7: Multi-step path ──────────────────────────────
    print('\n[8/8] Generating multi-step path GIF...')
    anchor_mus = []
    for cls_name in MULTI_STEP_CLASSES:
        print(f'  Finding centroid: {cls_name}...')
        _, mu, _ = find_centroid_sample(test_loader, cls_name, class_names,
                                        device, model)
        anchor_mus.append(mu)

    save_mult = os.path.join(OUT_DIR, 'multi_step_path.gif')
    make_multistep_gif(model, anchor_mus, MULTI_STEP_CLASSES,
                      save_mult, n_steps_per_seg=10, device=device)

    # ── Done ──────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'All outputs saved to: {OUT_DIR}/')
    print(f'Files:')
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f'  {f:40s}  {size_kb:7.1f} KB')
    print(f'Done!')


if __name__ == '__main__':
    main()
