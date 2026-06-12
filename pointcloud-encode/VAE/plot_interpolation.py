"""
Latent space interpolation visualization for VAE.

Two interpolation pairs:
  1. Same-class: two different chairs → sanity check
  2. Cross-class: airplane → lamp → dramatic transition

Output per pair: static 2×4 frame grid (PNG) + animated GIF.

Usage:
    cd pointcloud-encode/VAE
    python plot_interpolation.py \
        --checkpoint log/vae/2026-06-11_17-57/checkpoints/best_model.pth \
        --use_normals --gpu 0
"""
import os, sys, argparse, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import io

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PN2_DIR = os.path.join(THIS_DIR, '..', '..', '..', 'Pointnet_Pointnet2_pytorch')
sys.path.insert(0, PN2_DIR)
sys.path.insert(0, os.path.join(PN2_DIR, 'models'))
sys.path.insert(0, os.path.join(PN2_DIR, 'data_utils'))

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from pointnet2_vae import get_model

OUT_DIR = os.path.join(THIS_DIR, 'log', 'vae', 'analysis')

# Interpolation pairs: (class_a, class_b, label)
PAIRS = [
    ('chair', 'chair', 'same_class'),
    ('airplane', 'lamp', 'cross_class'),
]
N_FRAMES = 8  # interpolation steps


def parse_args():
    p = argparse.ArgumentParser('VAE Latent Space Interpolation')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--num_point', type=int, default=1024)
    p.add_argument('--num_category', type=int, default=40)
    p.add_argument('--use_normals', action='store_true', default=False)
    p.add_argument('--use_cpu', action='store_true', default=False)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--use_uniform_sample', action='store_true', default=False)
    p.add_argument('--process_data', action='store_true', default=False)
    return p.parse_args()


def find_samples(dataloader, class_a, class_b, class_names, device, model):
    """Find one sample each of class_a and class_b."""
    found_a = found_b = None
    label_a = class_names.index(class_a)
    label_b = class_names.index(class_b)

    for points, labels in dataloader:
        for i in range(points.size(0)):
            lbl = labels[i].item()
            if lbl == label_a and found_a is None:
                found_a = points[i:i+1].clone()
            if lbl == label_b and found_b is None:
                found_b = points[i:i+1].clone()
            if found_a is not None and found_b is not None:
                return found_a.to(device), found_b.to(device), label_a, label_b
    raise RuntimeError(f'Could not find samples for {class_a}/{class_b}')


def interpolate(model, pts_a, pts_b, n_frames):
    """Encode two point clouds, interpolate their mu vectors, decode each step."""
    model.eval()
    with torch.no_grad():
        # Normalize input to [B, C, N]
        if pts_a.dim() == 2:
            pts_a = pts_a.unsqueeze(0)
        if pts_b.dim() == 2:
            pts_b = pts_b.unsqueeze(0)

        pta = pts_a.transpose(2, 1)  # [1, C, N]
        ptb = pts_b.transpose(2, 1)

        mu_a = model.fc_mu(model.encode(pta))  # [1, 256]
        mu_b = model.fc_mu(model.encode(ptb))

        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            z_t = (1 - t) * mu_a + t * mu_b  # [1, 256]
            recon = model.decode(z_t)          # [1, M, 3]
            frames.append((t, recon.squeeze(0).cpu().numpy()))

    return frames


def make_plots(frames, class_a, class_b, label_a, label_b, pair_name, save_static, save_gif):
    """Generate static grid PNG and animated GIF."""
    n = len(frames)
    ncols = n
    nrows = 1

    # --- Determine consistent axis limits ---
    all_pts = np.concatenate([f[1] for f in frames], axis=0)
    lim = np.abs(all_pts).max() * 1.1

    # Choose colormap from blue (t=0) to red (t=1)
    cmap = plt.cm.coolwarm  # blue → red

    # --- Static figure: 2 rows × 4 cols ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten()

    for i, (t, pts) in enumerate(frames):
        ax = axes[i]
        color = cmap(t)  # interpolate color
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[color], s=2, alpha=0.85, edgecolors='none')
        ax.set_title(f't = {t:.2f}', fontsize=11, fontweight='bold',
                     color='#333333')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.view_init(elev=25, azim=60)

    start_label = class_a.replace('_', ' ')
    end_label = class_b.replace('_', ' ')
    fig.suptitle(f'Latent Space Interpolation: {start_label} → {end_label}',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_static, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Static: {os.path.basename(save_static)}')

    # --- Rotating GIF: camera orbits each interpolation step ---
    N_ANGLES = 15  # frames per t-step (15 = ~24° per step, smooth rotation)
    gif_frames = []
    start_label = class_a.replace('_', ' ')
    end_label = class_b.replace('_', ' ')

    for i, (t, pts) in enumerate(frames):
        for j, azim in enumerate(np.linspace(0, 360, N_ANGLES, endpoint=False)):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                                   subplot_kw={'projection': '3d'},
                                   facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            color = cmap(t)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=[color], s=4, alpha=0.9, edgecolors='none')
            ax.set_title(f'{start_label} → {end_label}    t = {t:.2f}',
                         fontsize=13, fontweight='bold', color='#e0e0e0')
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.view_init(elev=25, azim=azim)
            # Hide panes for cleaner look
            ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#333355')
            ax.yaxis.pane.set_edgecolor('#333355')
            ax.zaxis.pane.set_edgecolor('#333355')
            ax.grid(False)
            fig.tight_layout(pad=0)

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                       facecolor='#1a1a2e')
            buf.seek(0)
            img = imageio.v2.imread(buf)
            gif_frames.append(img)
            plt.close(fig)
            buf.close()

        # ---- Transition card between t-steps ----
        # Add 3 identical frames with t-label to pause between rotations
        for _ in range(3):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            ax.text(0.5, 0.5, f't = {t:.2f}', fontsize=42, fontweight='bold',
                    ha='center', va='center', color='#e0e0e0',
                    transform=ax.transAxes)
            ax.text(0.5, 0.35, f'{start_label} → {end_label}',
                    fontsize=16, ha='center', va='center', color='#888888',
                    transform=ax.transAxes)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('off')
            fig.tight_layout(pad=0)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                       facecolor='#1a1a2e')
            buf.seek(0)
            img = imageio.v2.imread(buf)
            gif_frames.append(img)
            plt.close(fig)
            buf.close()

    # Save GIF: 120ms per rotation frame, loop forever
    # Resize all frames to consistent shape
    target_h, target_w = gif_frames[0].shape[:2]
    resized = []
    for img in gif_frames:
        if img.shape[:2] != (target_h, target_w):
            import PIL.Image
            pil_img = PIL.Image.fromarray(img)
            pil_img = pil_img.resize((target_w, target_h), PIL.Image.LANCZOS)
            img = np.array(pil_img)
        resized.append(img)
    imageio.mimsave(save_gif, resized, duration=80, loop=0)
    total = len(gif_frames)
    print(f'  GIF:    {os.path.basename(save_gif)} '
          f'({total} frames, ~{total * 80 / 1000:.1f}s)')

    # --- Fixed-angle GIF: single viewpoint, 8-step transition ---
    save_gif_fixed = save_gif.replace('.gif', '_fixed.gif')
    fixed_frames = []
    for i, (t, pts) in enumerate(frames):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                               subplot_kw={'projection': '3d'},
                               facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        color = cmap(t)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[color], s=4, alpha=0.9, edgecolors='none')
        ax.set_title(f'{start_label} → {end_label}    t = {t:.2f}',
                     fontsize=13, fontweight='bold', color='#e0e0e0')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.view_init(elev=25, azim=45)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333355')
        ax.yaxis.pane.set_edgecolor('#333355')
        ax.zaxis.pane.set_edgecolor('#333355')
        ax.grid(False)
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor='#1a1a2e')
        buf.seek(0)
        img = imageio.v2.imread(buf)
        fixed_frames.append(img)
        plt.close(fig)
        buf.close()

    imageio.mimsave(save_gif_fixed, fixed_frames, duration=800, loop=0)
    print(f'  GIF:    {os.path.basename(save_gif_fixed)} '
          f'({len(fixed_frames)} frames)')


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
    test_ds = ModelNetDataLoader(root=data_path, args=args, split='test',
                                  process_data=args.process_data)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    class_names = test_ds.cat

    # ---- Load model ----
    print(f'Loading VAE from: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vae_args = ckpt.get('args', None)
    model = get_model(
        latent_dim=vae_args.latent_dim if vae_args else 256,
        normal_channel=vae_args.use_normals if vae_args else args.use_normals,
        num_output_points=vae_args.num_output_points if vae_args else 1024,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Epoch: {ckpt.get("epoch", -1)}, Test CD: {ckpt.get("test_cd", -1):.6f}')

    # ---- Run interpolations ----
    for class_a, class_b, pair_name in PAIRS:
        print(f'\n=== {pair_name}: {class_a} → {class_b} ===')

        print(f'  Finding samples...')
        pts_a, pts_b, la, lb = find_samples(
            test_loader, class_a, class_b, class_names, device, model)

        print(f'  Interpolating {N_FRAMES} steps...')
        frames = interpolate(model, pts_a, pts_b, N_FRAMES)

        save_static = os.path.join(OUT_DIR, f'interpolation_{pair_name}.png')
        save_gif = os.path.join(OUT_DIR, f'interpolation_{pair_name}.gif')
        make_plots(frames, class_a, class_b, la, lb, pair_name,
                   save_static, save_gif)

    print('\nDone! 2 static PNGs + 4 GIFs (2 rotating + 2 fixed) saved.')


if __name__ == '__main__':
    main()
