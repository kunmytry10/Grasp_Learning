"""
Plot VAE training curves from log file.

Parses the training log and produces a 4-panel figure:
  1. CD Loss (Train + Test)
  2. KL Divergence (Train + Test)
  3. Beta annealing schedule
  4. Learning rate schedule

Usage:
    cd pointcloud-encode/VAE
    python plot_training_curves.py
"""
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, 'log', 'vae', '2026-06-11_17-57',
                         'logs', 'pointnet2_vae.txt')
OUT_DIR = os.path.join(LOG_DIR, 'log', 'vae', 'analysis')
OUT_FILE = os.path.join(OUT_DIR, 'training_curves.png')

# Hyperparameters from training config (must match train_vae.py defaults)
WARMUP_EPOCHS = 30
TOTAL_ANNEAL_EPOCHS = 50
MAX_BETA = 1.0
LR_INIT = 0.001
LR_STEP = 20
LR_GAMMA = 0.7

# ---------------------------------------------------------------------------
# Parse log
# ---------------------------------------------------------------------------

def parse_log(log_path):
    """Parse VAE training log into structured arrays."""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    epochs = []
    train_cd, train_kl = [], []
    test_cd, test_kl = [], []
    betas, lrs = [], []

    line_pattern = re.compile(
        r'Epoch \d+ \((\d+)/\d+\)\s+beta=([\d.]+)\s+lr=([\d.]+)')
    train_pattern = re.compile(
        r'Train CD: ([\d.]+)\s+KL: ([\d.]+)')
    test_pattern = re.compile(
        r'Test  CD: ([\d.]+)\s+KL: ([\d.]+)')

    i = 0
    while i < len(lines):
        line = lines[i]
        m = line_pattern.search(line)
        if m:
            epoch_num = int(m.group(1))
            epochs.append(epoch_num)
            betas.append(float(m.group(2)))
            lrs.append(float(m.group(3)))

            # Next line should be Train
            if i + 1 < len(lines):
                tm = train_pattern.search(lines[i + 1])
                if tm:
                    train_cd.append(float(tm.group(1)))
                    train_kl.append(float(tm.group(2)))
            # Next+1 line should be Test
            if i + 2 < len(lines):
                tsm = test_pattern.search(lines[i + 2])
                if tsm:
                    test_cd.append(float(tsm.group(1)))
                    test_kl.append(float(tsm.group(2)))
            i += 3
        else:
            i += 1

    return (np.array(epochs), np.array(train_cd), np.array(train_kl),
            np.array(test_cd), np.array(test_kl),
            np.array(betas), np.array(lrs))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_curves(epochs, train_cd, train_kl, test_cd, test_kl, betas, lrs):
    """4-panel training curves figure."""
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_cd, ax_kl), (ax_beta, ax_lr) = axes

    # ---- Common styling ----
    # Highlight beta annealing zone
    anneal_start = WARMUP_EPOCHS
    anneal_end = WARMUP_EPOCHS + TOTAL_ANNEAL_EPOCHS
    for ax in [ax_cd, ax_kl]:
        ax.axvspan(anneal_start, anneal_end, alpha=0.08, color='orange',
                   label=u'β annealing (30–80)')

    # ---- Subplot 1: CD Loss ----
    ax_cd.plot(epochs, train_cd, 'b-', linewidth=1.2, alpha=0.8,
               label='Train CD')
    ax_cd.plot(epochs, test_cd, 'r-', linewidth=1.2, alpha=0.9,
               label='Test CD')
    # Mark best test CD
    best_idx = np.argmin(test_cd)
    ax_cd.scatter(epochs[best_idx], test_cd[best_idx],
                  c='red', s=60, zorder=5, edgecolors='darkred', linewidths=1)
    ax_cd.annotate(f'Best: {test_cd[best_idx]:.4f}\n(epoch {epochs[best_idx]})',
                   xy=(epochs[best_idx], test_cd[best_idx]),
                   xytext=(epochs[best_idx] + 12, test_cd[best_idx] + 0.03),
                   fontsize=8, color='darkred', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2))
    ax_cd.set_ylabel('Chamfer Distance', fontsize=12)
    ax_cd.set_title('CD Loss (Reconstruction)', fontsize=13, fontweight='bold')
    ax_cd.legend(fontsize=9, loc='upper right')
    ax_cd.grid(alpha=0.25)
    ax_cd.set_xlim(0, len(epochs) + 5)

    # ---- Subplot 2: KL Divergence ----
    ax_kl.plot(epochs, train_kl, 'b-', linewidth=1.2, alpha=0.8,
               label='Train KL')
    ax_kl.plot(epochs, test_kl, 'r-', linewidth=1.2, alpha=0.9,
               label='Test KL')
    ax_kl.set_ylabel('KL Divergence (normalized)', fontsize=12)
    ax_kl.set_xlabel('Epoch', fontsize=12)
    ax_kl.set_title('KL Divergence (Regularization)', fontsize=13, fontweight='bold')
    ax_kl.legend(fontsize=9, loc='upper right')
    ax_kl.grid(alpha=0.25)
    ax_kl.set_xlim(0, len(epochs) + 5)

    # ---- Subplot 3: Beta Annealing ----
    ax_beta.plot(epochs, betas, 'g-', linewidth=2)
    # Also plot expected theoretical beta
    theoretical = np.zeros(len(epochs))
    for i, e in enumerate(epochs):
        if e < WARMUP_EPOCHS:
            theoretical[i] = 0
        elif e < WARMUP_EPOCHS + TOTAL_ANNEAL_EPOCHS:
            theoretical[i] = (e - WARMUP_EPOCHS) / TOTAL_ANNEAL_EPOCHS * MAX_BETA
        else:
            theoretical[i] = MAX_BETA
    ax_beta.plot(epochs, theoretical, '--', color='gray', linewidth=1,
                 alpha=0.6, label='Theoretical')
    ax_beta.set_ylabel('β', fontsize=12)
    ax_beta.set_xlabel('Epoch', fontsize=12)
    ax_beta.set_title('KL Weight (Beta Annealing)', fontsize=13, fontweight='bold')
    ax_beta.legend(fontsize=9)
    ax_beta.grid(alpha=0.25)
    ax_beta.set_ylim(-0.05, 1.15)
    ax_beta.set_xlim(0, len(epochs) + 5)
    # Annotate phases
    ax_beta.text(WARMUP_EPOCHS / 2, 0.08, 'Pure AE\nβ=0', ha='center',
                 fontsize=9, color='gray', fontstyle='italic')
    ax_beta.text(WARMUP_EPOCHS + TOTAL_ANNEAL_EPOCHS / 2, 0.5, 'Annealing\nβ↑',
                 ha='center', fontsize=9, color='orange', fontstyle='italic')
    ax_beta.text(WARMUP_EPOCHS + TOTAL_ANNEAL_EPOCHS + 60, 0.92, 'Full VAE\nβ=1',
                 ha='center', fontsize=9, color='green', fontstyle='italic')

    # ---- Subplot 4: Learning Rate ----
    ax_lr.plot(epochs, lrs, 'purple', linewidth=1.5)
    ax_lr.set_ylabel('Learning Rate', fontsize=12)
    ax_lr.set_xlabel('Epoch', fontsize=12)
    ax_lr.set_title('Learning Rate (StepLR, step=20, γ=0.7)',
                    fontsize=13, fontweight='bold')
    ax_lr.grid(alpha=0.25)
    ax_lr.set_xlim(0, len(epochs) + 5)
    # Annotate LR drops
    for step in range(LR_STEP, len(epochs) + 1, LR_STEP):
        if step <= len(epochs):
            lr_val = LR_INIT * (LR_GAMMA ** (step // LR_STEP))
            ax_lr.axvline(x=step, color='gray', linestyle=':', alpha=0.4,
                          linewidth=0.8)
            ax_lr.text(step + 1, lr_val * 1.3, f'{lr_val:.6f}',
                       fontsize=7, color='gray', rotation=90, va='bottom')

    # ---- Global ----
    fig.suptitle('PointNet++ VAE Training — 200 Epochs on ModelNet40',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f'Saved: {OUT_FILE}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Parsing log: {LOG_FILE}')
    epochs, train_cd, train_kl, test_cd, test_kl, betas, lrs = \
        parse_log(LOG_FILE)

    print(f'Parsed {len(epochs)} epochs')
    print(f'  Train CD: {train_cd[0]:.4f} → {train_cd[-1]:.4f}')
    print(f'  Test  CD: {test_cd[0]:.4f} → {test_cd[-1]:.4f}')
    print(f'  Best  CD: {test_cd.min():.6f} @ epoch {epochs[test_cd.argmin()]}')
    print(f'  Train KL: {train_kl[0]:.4f} → {train_kl[-1]:.4f}')
    print(f'  Test  KL: {test_kl[0]:.4f} → {test_kl[-1]:.4f}')
    print(f'  Beta:     {betas[0]:.4f} → {betas[-1]:.4f}')
    print(f'  LR:       {lrs[0]:.6f} → {lrs[-1]:.6f}')

    plot_curves(epochs, train_cd, train_kl, test_cd, test_kl, betas, lrs)


if __name__ == '__main__':
    main()
