"""
Train PointNet++ VAE on ModelNet40 (self-supervised).

Usage:
    cd pointcloud-encode/VAE
    python train_vae.py --batch_size 24 --epoch 200 --use_normals
"""
import os
import sys
import torch
import numpy as np
import datetime
import logging
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm

# Paths to PointNet++ project (external dependencies)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PN2_DIR = os.path.join(THIS_DIR, '..', '..', 'Pointnet_Pointnet2_pytorch')
sys.path.insert(0, PN2_DIR)                           # for provider
sys.path.insert(0, os.path.join(PN2_DIR, 'models'))   # for pointnet2_utils

import provider
from data_utils.ModelNetDataLoader import ModelNetDataLoader


def parse_args():
    parser = argparse.ArgumentParser('VAE training')
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--model', default='pointnet2_vae')
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--use_normals', action='store_true', default=False)
    parser.add_argument('--process_data', action='store_true', default=False)
    parser.add_argument('--use_uniform_sample', action='store_true', default=False)
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40])

    # VAE-specific args
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_output_points', type=int, default=1024)
    parser.add_argument('--warmup_epochs', type=int, default=30,
                        help='epochs with beta=0 (pure AE)')
    parser.add_argument('--total_anneal_epochs', type=int, default=50,
                        help='epochs to linearly anneal beta 0->1')
    parser.add_argument('--max_beta', type=float, default=1.0)

    return parser.parse_args()


def compute_beta(epoch, warmup_epochs, total_anneal_epochs, max_beta):
    """Linear beta annealing: 0 -> max_beta."""
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + total_anneal_epochs:
        progress = (epoch - warmup_epochs) / total_anneal_epochs
        return progress * max_beta
    else:
        return max_beta


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, loader, criterion, args):
    """Evaluate reconstruction Chamfer Distance on test set."""
    model.eval()
    total_cd = 0.0
    total_kl = 0.0
    num_samples = 0

    with torch.no_grad():
        for points, _ in tqdm(loader, total=len(loader), desc='Eval'):
            if not args.use_cpu:
                points = points.cuda()
            points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

            result = model(points)
            loss_dict = criterion(result, beta=1.0)

            total_cd += loss_dict['recon'].item() * points.size(0)
            total_kl += loss_dict['kl'].item() * points.size(0)
            num_samples += points.size(0)

    return total_cd / num_samples, total_kl / num_samples


def main(args):
    def log_string(msg):
        logger.info(msg)
        print(msg)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ---- Directory setup ----
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('vae')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # ---- Logging ----
    logger = logging.getLogger("VAE")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # ---- Data loading ----
    log_string('Load dataset ...')
    data_path = os.path.join(PN2_DIR, 'data', 'modelnet40_normal_resampled')

    train_dataset = ModelNetDataLoader(
        root=data_path, args=args, split='train',
        process_data=args.process_data)
    test_dataset = ModelNetDataLoader(
        root=data_path, args=args, split='test',
        process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=10)

    # ---- Model loading ----
    log_string('Load model ...')
    model_lib = importlib.import_module(args.model)

    # Copy source files to experiment dir for reproducibility
    model_src = os.path.join(THIS_DIR, '%s.py' % args.model)
    if os.path.exists(model_src):
        shutil.copy(model_src, str(exp_dir))
    utils_src = os.path.join(PN2_DIR, 'models', 'pointnet2_utils.py')
    if os.path.exists(utils_src):
        shutil.copy(utils_src, str(exp_dir))
    shutil.copy(__file__, str(exp_dir))

    classifier = model_lib.get_model(
        latent_dim=args.latent_dim,
        normal_channel=args.use_normals,
        num_output_points=args.num_output_points)
    criterion = model_lib.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # ---- Checkpoint resume ----
    try:
        checkpoint = torch.load(
            str(checkpoints_dir) + '/best_model.pth', weights_only=False)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrained model from epoch %d' % start_epoch)
    except Exception:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    # ---- Optimizer ----
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_cd = float('inf')

    # ---- Training ----
    log_string('Start training...')
    for epoch in range(start_epoch, args.epoch):
        beta = compute_beta(
            epoch, args.warmup_epochs, args.total_anneal_epochs, args.max_beta)
        log_string(
            'Epoch %d (%d/%s)  beta=%.4f  lr=%.6f' %
            (global_epoch + 1, epoch + 1, args.epoch, beta,
             optimizer.param_groups[0]['lr']))

        classifier = classifier.train()
        train_cd_list = []
        train_kl_list = []

        for batch_id, (points, _) in tqdm(
                enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                smoothing=0.9, desc='Train'):
            optimizer.zero_grad()

            # ---- Data augmentation (same as classification) ----
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(
                points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(
                points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)  # [B, N, C] -> [B, C, N]

            if not args.use_cpu:
                points = points.cuda()

            result = classifier(points)
            loss_dict = criterion(result, beta=beta)

            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 10.0)
            optimizer.step()
            global_step += 1

            train_cd_list.append(loss_dict['recon'].item())
            train_kl_list.append(loss_dict['kl'].item())

        train_cd = np.mean(train_cd_list)
        train_kl = np.mean(train_kl_list)
        log_string(
            'Train CD: %.6f  KL: %.6f' % (train_cd, train_kl))

        # ---- Evaluation ----
        with torch.no_grad():
            test_cd, test_kl = test(
                classifier.eval(), testDataLoader, criterion, args)
            log_string(
                'Test  CD: %.6f  KL: %.6f' % (test_cd, test_kl))

            if test_cd < best_cd:
                best_cd = test_cd
                best_epoch = epoch + 1
                log_string('New best CD! Saving...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                state = {
                    'epoch': best_epoch,
                    'test_cd': test_cd,
                    'test_kl': test_kl,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                }
                torch.save(state, savepath)

            log_string(
                'Best CD: %.6f (epoch %d)' % (best_cd, best_epoch))

        scheduler.step()
        global_epoch += 1

    log_string('End of training. Best CD: %.6f' % best_cd)


if __name__ == '__main__':
    args = parse_args()
    main(args)
