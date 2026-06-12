"""
Extract VAE encoder features from ModelNet40 test set.

Usage:
    cd pointcloud-encode/VAE
    python eval_vae.py --checkpoint log/vae/.../best_model.pth --use_normals

Output:
    features.npy  [N, latent_dim]
    labels.npy    [N,]
"""
import os
import sys
import torch
import numpy as np
import argparse
import importlib

from pathlib import Path
from tqdm import tqdm

# Paths to PointNet++ project (external dependencies)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PN2_DIR = os.path.join(THIS_DIR, '..', '..', '..', 'Pointnet_Pointnet2_pytorch')
sys.path.insert(0, PN2_DIR)
sys.path.insert(0, os.path.join(PN2_DIR, 'models'))

from data_utils.ModelNetDataLoader import ModelNetDataLoader


def parse_args():
    parser = argparse.ArgumentParser('VAE feature extraction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to best_model.pth')
    parser.add_argument('--save_dir', type=str,
                        default='log/vae/features/')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--use_normals', action='store_true', default=False)
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--use_uniform_sample', action='store_true',
                        default=False)
    parser.add_argument('--process_data', action='store_true', default=False)
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ---- Load checkpoint ----
    print('Loading checkpoint: %s' % args.checkpoint)
    checkpoint = torch.load(args.checkpoint, weights_only=False)

    # Determine model config from checkpoint
    saved_args = checkpoint.get('args', None)
    if saved_args is not None:
        latent_dim = saved_args.latent_dim
        normal_channel = saved_args.use_normals
        num_output_points = saved_args.num_output_points
        model_name = saved_args.model
    else:
        # Fallback defaults
        latent_dim = 256
        normal_channel = args.use_normals
        num_output_points = args.num_point
        model_name = 'pointnet2_vae'

    print('Model config: latent_dim=%d, normals=%s, points=%d' %
          (latent_dim, normal_channel, num_output_points))

    # ---- Load model ----
    model_lib = importlib.import_module(model_name)
    model = model_lib.get_model(
        latent_dim=latent_dim,
        normal_channel=normal_channel,
        num_output_points=num_output_points)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if not args.use_cpu:
        model = model.cuda()

    print('Model loaded. Epoch: %d, Test CD: %.6f' %
          (checkpoint.get('epoch', -1), checkpoint.get('test_cd', -1)))

    # ---- Data loading ----
    data_path = os.path.join(PN2_DIR, 'data', 'modelnet40_normal_resampled')
    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split='test',
        process_data=args.process_data)
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=10)

    # ---- Extract features ----
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for points, labels in tqdm(testDataLoader, desc='Extracting'):
            if not args.use_cpu:
                points = points.cuda()
            points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

            global_feat = model.encode(points)    # [B, 1024]
            mu = model.fc_mu(global_feat)          # [B, latent_dim]

            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_mu, axis=0)       # [N, latent_dim]
    labels = np.concatenate(all_labels, axis=0)      # [N,]

    # ---- Save ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    feat_path = save_dir / 'features.npy'
    label_path = save_dir / 'labels.npy'
    np.save(str(feat_path), features)
    np.save(str(label_path), labels)

    print('Saved features: %s  shape=%s' % (feat_path, features.shape))
    print('Saved labels:   %s  shape=%s' % (label_path, labels.shape))
    print('Done.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
