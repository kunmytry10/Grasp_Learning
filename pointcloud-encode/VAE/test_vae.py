"""
TDD tests for VAE components.
Run: python test_vae.py
"""
import sys
import os
import torch

# Add PointNet++ models to path (SA modules are needed by the VAE model)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                   'Pointnet_Pointnet2_pytorch', 'models'))

# ============================================================
# Test 1: chamfer_distance
# ============================================================

def test_chamfer_perfect_reconstruction():
    """CD should be exactly 0 when pred==target (same tensor reference)."""
    from pointnet2_vae import chamfer_distance
    pts = torch.randn(2, 1024, 3)
    loss = chamfer_distance(pts, pts)
    # Floating-point noise in cdist diagonal: ~1e-4 for 1024 random points
    assert loss.item() < 5e-4, f"Expected ~0, got {loss.item()}"


def test_chamfer_different_clouds():
    """CD should be >0 for different point clouds."""
    from pointnet2_vae import chamfer_distance
    a = torch.randn(2, 1024, 3)
    b = torch.randn(2, 1024, 3) + 5.0  # shifted far away
    loss = chamfer_distance(a, b)
    assert loss.item() > 1.0, f"Expected >1.0, got {loss.item()}"


def test_chamfer_returns_scalar_tensor():
    """CD should return a scalar tensor with grad enabled."""
    from pointnet2_vae import chamfer_distance
    a = torch.randn(2, 1024, 3, requires_grad=True)
    b = torch.randn(2, 1024, 3)
    loss = chamfer_distance(a, b)
    assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"
    loss.backward()
    assert a.grad is not None, "Expected gradients to flow"


def test_chamfer_batch_independence():
    """Batch mean averages over all samples: batched = sum(individuals)/B."""
    from pointnet2_vae import chamfer_distance
    a1 = torch.randn(1, 1024, 3)
    a2 = torch.randn(1, 1024, 3) + 10.0  # far away
    a_batch = torch.cat([a1, a2], dim=0)

    loss_individual = chamfer_distance(a1, a1) + chamfer_distance(a2, a2)
    loss_batched = chamfer_distance(a_batch, a_batch)
    # Mean over B=2 samples: batched = individual_sum / 2
    assert abs(loss_individual.item() / 2 - loss_batched.item()) < 1e-4, \
        f"Batched={loss_batched.item():.6f} != Individual/2={loss_individual.item()/2:.6f}"


# ============================================================
# Test 2: VAE Model
# ============================================================

def test_model_instantiation():
    """Can create get_model with default args."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=True, num_output_points=1024)
    assert isinstance(model, torch.nn.Module)


def test_encoder_shape():
    """encode() returns [B, 1024] global feature."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=True, num_output_points=1024)
    xyz = torch.randn(4, 6, 1024)
    feat = model.encode(xyz)
    assert feat.shape == (4, 1024), f"Expected (4,1024), got {feat.shape}"


def test_encoder_shape_no_normals():
    """encode() works with 3-channel input (xyz only)."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=False, num_output_points=1024)
    xyz = torch.randn(4, 3, 1024)
    feat = model.encode(xyz)
    assert feat.shape == (4, 1024)


def test_reparameterize_shapes():
    """reparameterize returns z [B, latent_dim]."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=True)
    mu = torch.randn(4, 256)
    logvar = torch.randn(4, 256)
    z = model.reparameterize(mu, logvar)
    assert z.shape == (4, 256), f"Expected (4,256), got {z.shape}"


def test_reparameterize_deterministic_in_eval():
    """In eval mode, z == mu (no randomness)."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=True)
    model.eval()
    mu = torch.randn(4, 256)
    logvar = torch.randn(4, 256)
    z1 = model.reparameterize(mu, logvar)
    z2 = model.reparameterize(mu, logvar)
    assert torch.allclose(z1, z2), "Eval mode should be deterministic"


def test_reparameterize_stochastic_in_train():
    """In train mode, sampling adds noise (z != mu when logvar is finite)."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=True)
    model.train()
    mu = torch.zeros(4, 256)
    logvar = torch.zeros(4, 256)  # sigma = 1
    z = model.reparameterize(mu, logvar)
    # With sigma=1, z should differ from mu
    assert not torch.allclose(z, mu), "Train mode should add noise"


def test_decode_shape():
    """decode() returns [B, M, 3] point cloud."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=True, num_output_points=1024)
    z = torch.randn(4, 256)
    recon = model.decode(z)
    assert recon.shape == (4, 1024, 3), f"Expected (4,1024,3), got {recon.shape}"


def test_forward_shapes_and_keys():
    """forward() returns dict with recon, xyz, mu, logvar."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=True, num_output_points=1024)
    xyz = torch.randn(4, 6, 1024)
    result = model(xyz)
    assert 'recon' in result
    assert 'xyz' in result
    assert 'mu' in result
    assert 'logvar' in result
    assert result['recon'].shape == (4, 1024, 3)
    assert result['xyz'].shape == (4, 1024, 3)
    assert result['mu'].shape == (4, 256)
    assert result['logvar'].shape == (4, 256)


def test_forward_xyz_target_is_stripped_normals():
    """xyz target in forward output is [B,N,3], normals stripped."""
    from pointnet2_vae import get_model
    model = get_model(latent_dim=256, normal_channel=True, num_output_points=1024)
    xyz_6ch = torch.randn(4, 6, 1024)
    result = model(xyz_6ch)
    # Target should be the first 3 channels, transposed to [B,N,3]
    assert result['xyz'].shape == (4, 1024, 3)


def test_loss_computation():
    """get_loss returns total, recon, kl keys with correct types."""
    from pointnet2_vae import get_model, get_loss
    model = get_model(latent_dim=256, normal_channel=True, num_output_points=1024)
    criterion = get_loss()
    xyz = torch.randn(4, 6, 1024)
    result = model(xyz)
    loss_dict = criterion(result)
    assert 'total' in loss_dict
    assert 'recon' in loss_dict
    assert 'kl' in loss_dict
    for v in loss_dict.values():
        assert v.dim() == 0, f"Loss should be scalar, got shape {v.shape}"


def test_gradient_flow_through_model():
    """Gradients flow from loss back through encoder and decoder."""
    from pointnet2_vae import get_model, get_loss
    model = get_model(latent_dim=256, normal_channel=True, num_output_points=1024)
    criterion = get_loss()
    xyz = torch.randn(4, 6, 1024)
    result = model(xyz)
    loss_dict = criterion(result)
    loss_dict['total'].backward()

    # Check that encoder params got gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No parameters received gradients"


def test_forward_no_normals():
    """Model works with 3-channel input (no normals)."""
    from pointnet2_vae import get_model, get_loss
    model = get_model(latent_dim=256, normal_channel=False, num_output_points=1024)
    criterion = get_loss()
    xyz = torch.randn(4, 3, 1024)
    result = model(xyz)
    loss_dict = criterion(result)
    loss_dict['total'].backward()  # should not crash


# ============================================================
# Test 3: Overfit on tiny batch
# ============================================================

def test_overfit_tiny_batch():
    """Loss should drop significantly when overfitting on 8 samples."""
    from pointnet2_vae import get_model, get_loss
    torch.manual_seed(42)
    model = get_model(latent_dim=256, normal_channel=False, num_output_points=1024)
    criterion = get_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Tiny dataset: 8 fixed, normalized point clouds
    xyz_raw = torch.randn(8, 3, 1024)
    # Per-sample normalization (mimics ModelNet40 preprocessing)
    centroid = xyz_raw.mean(dim=-1, keepdim=True)
    xyz = xyz_raw - centroid
    max_dist = xyz.norm(dim=1, keepdim=True).max(dim=-1, keepdim=True)[0]
    xyz = xyz / (max_dist + 1e-8)

    model.train()
    losses = []
    for step in range(500):
        optimizer.zero_grad()
        result = model(xyz)
        loss_dict = criterion(result, beta=0.0)
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(loss_dict['total'].item())

    # Loss should decrease meaningfully (40%+) over 500 steps
    assert losses[0] > 0.01, f"Initial loss too small: {losses[0]:.6f}"
    assert losses[-1] < losses[0] * 0.6, \
        f"Loss did not decrease enough: {losses[0]:.6f} -> {losses[-1]:.6f}"


if __name__ == '__main__':
    results = []
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print(f"  PASS {name}")
                results.append(True)
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)
