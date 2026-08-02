from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_bound_log_depth_residual(
    residual: torch.Tensor,
    max_abs: float | None,
) -> torch.Tensor:
    """
    Smoothly constrain an SSR log-depth update without changing its local scale.

    ``limit * tanh(raw / limit)`` is first-order identical to ``raw`` around
    zero, remains differentiable, and prevents one refinement cycle from
    moving a point by more than ``limit`` in log-depth. Non-finite raw values
    deliberately remain non-finite so that the training safety checks cannot
    be hidden by the bound.
    """
    if max_abs is None or float(max_abs) == 0.0:
        return residual
    limit = float(max_abs)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("Log-depth residual bound must be finite and non-negative")
    bounded = limit * torch.tanh(residual / limit)
    return torch.where(torch.isfinite(residual), bounded, residual)


def factorize_points(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert (..., X, Y, Z) points to (..., X/Z, Y/Z, log Z)."""
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points[..., 3], got {tuple(points.shape)}")
    z = points[..., 2:3].clamp_min(eps)
    return torch.cat((points[..., :2] / z, z.log()), dim=-1)


def unfactorize_points(factorized: torch.Tensor) -> torch.Tensor:
    """Convert (..., u, v, log Z) back to Euclidean camera-space points."""
    if factorized.shape[-1] != 3:
        raise ValueError(f"Expected factorized[..., 3], got {tuple(factorized.shape)}")
    depth = factorized[..., 2:3].exp()
    return torch.cat((factorized[..., :2] * depth, depth), dim=-1)


def _ceil_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


@dataclass
class VoxelizedShell:
    """Sparse shell and the metadata required to map it back to image pixels."""

    coordinates: torch.Tensor
    features: torch.Tensor
    spatial_shape: Tuple[int, int, int]
    batch_size: int
    image_size: Tuple[int, int]
    logical_depth: torch.Tensor
    depth_offsets: torch.Tensor

    def statistics(self) -> Dict[str, torch.Tensor]:
        depth_span = self.logical_depth.amax(dim=(-2, -1)) - self.logical_depth.amin(dim=(-2, -1)) + 1
        return {
            "active_voxels": torch.tensor(
                self.coordinates.shape[0], device=self.features.device, dtype=torch.long
            ),
            "depth_span": depth_span,
            "depth_offsets": self.depth_offsets,
        }


def voxelize_factorized(
    factorized: torch.Tensor,
    voxel_resolution: float = 200.0,
    num_downsamples: int = 4,
) -> VoxelizedShell:
    """
    Voxelize q=(u,v,zeta) using c=(row,col,round(D*zeta)).

    Logical coordinates follow the paper. Storage coordinates follow spconv's
    [batch, depth, row, col] convention and are shifted per sample so that every
    coordinate is non-negative. Translation does not change sparse adjacency.
    """
    if factorized.ndim != 4 or factorized.shape[-1] != 3:
        raise ValueError(f"Expected [B,H,W,3], got {tuple(factorized.shape)}")
    if not torch.isfinite(factorized).all():
        raise ValueError("SSR factorized coordinates must be finite")

    batch_size, height, width, _ = factorized.shape
    device = factorized.device
    logical_depth = torch.round(voxel_resolution * factorized[..., 2]).to(torch.long)
    depth_offsets = logical_depth.amin(dim=(-2, -1))
    storage_depth = logical_depth - depth_offsets[:, None, None]

    rows, cols = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.long),
        torch.arange(width, device=device, dtype=torch.long),
        indexing="ij",
    )
    rows = rows.unsqueeze(0).expand(batch_size, -1, -1)
    cols = cols.unsqueeze(0).expand(batch_size, -1, -1)
    batches = torch.arange(batch_size, device=device, dtype=torch.long)[:, None, None]
    batches = batches.expand(-1, height, width)

    coordinates = torch.stack((batches, storage_depth, rows, cols), dim=-1)
    coordinates = coordinates.reshape(-1, 4).to(torch.int32).contiguous()
    features = factorized.reshape(-1, 3).contiguous()

    stride = 2**num_downsamples
    max_depth = int(storage_depth.amax().item()) + 1
    spatial_shape = (
        _ceil_multiple(max_depth, stride),
        _ceil_multiple(height, stride),
        _ceil_multiple(width, stride),
    )
    return VoxelizedShell(
        coordinates=coordinates,
        features=features,
        spatial_shape=spatial_shape,
        batch_size=batch_size,
        image_size=(height, width),
        logical_depth=logical_depth,
        depth_offsets=depth_offsets,
    )


def _coordinate_hash(coordinates: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    z_size, y_size, x_size = (int(v) for v in spatial_shape)
    coordinates = coordinates.to(torch.long)
    batch, z, y, x = coordinates.unbind(dim=-1)
    return ((batch * z_size + z) * y_size + y) * x_size + x


def gather_features_at_coordinates(
    features: torch.Tensor,
    source_coordinates: torch.Tensor,
    target_coordinates: torch.Tensor,
    spatial_shape: Sequence[int],
) -> torch.Tensor:
    """Reorder sparse features to match a target coordinate order."""
    if torch.equal(source_coordinates, target_coordinates):
        return features
    source_hash = _coordinate_hash(source_coordinates, spatial_shape)
    target_hash = _coordinate_hash(target_coordinates, spatial_shape)
    sorted_hash, order = source_hash.sort()
    positions = torch.searchsorted(sorted_hash, target_hash)
    if positions.numel() and (
        (positions >= sorted_hash.numel()).any()
        or not torch.equal(sorted_hash[positions.clamp_max(sorted_hash.numel() - 1)], target_hash)
    ):
        raise RuntimeError("Sparse decoder did not recover every input voxel")
    return features[order[positions]]


def sample_visual_features_for_voxels(
    visual_features: torch.Tensor,
    coordinates: torch.Tensor,
    image_size: Tuple[int, int],
    level_scale: int,
) -> torch.Tensor:
    """Sample by (batch,row,col); voxel depth intentionally has no effect."""
    height, width = image_size
    target_height = max(1, (height + level_scale - 1) // level_scale)
    target_width = max(1, (width + level_scale - 1) // level_scale)
    visual = F.interpolate(
        visual_features.float(),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    batch, _, row, col = coordinates.to(torch.long).unbind(dim=-1)
    row = row.clamp_max(target_height - 1)
    col = col.clamp_max(target_width - 1)
    return visual[batch, :, row, col]


SSR_NORMALIZATIONS = ("batch_norm", "group_norm", "layer_norm")


def sparse_feature_normalization(
    channels: int,
    normalization: str,
) -> nn.Module:
    """Build a normalization that operates on sparse features shaped [N, C]."""
    if normalization == "batch_norm":
        return nn.BatchNorm1d(channels)
    if normalization == "layer_norm":
        return nn.LayerNorm(channels)
    if normalization == "group_norm":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(
        f"Unsupported SSR normalization {normalization!r}; "
        f"expected one of {SSR_NORMALIZATIONS}"
    )


class _SparseResidualBlock(nn.Module):
    def __init__(
        self,
        spconv,
        channels: int,
        indice_key: str,
        conv_algo,
        normalization: str,
    ):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            channels,
            channels,
            3,
            padding=1,
            bias=False,
            indice_key=f"{indice_key}_1",
            algo=conv_algo,
        )
        self.conv2 = spconv.SubMConv3d(
            channels,
            channels,
            3,
            padding=1,
            bias=False,
            indice_key=f"{indice_key}_2",
            algo=conv_algo,
        )
        self.norm1 = sparse_feature_normalization(channels, normalization)
        self.norm2 = sparse_feature_normalization(channels, normalization)

    def forward(self, sparse):
        identity = sparse.features
        out = sparse.replace_feature(F.relu(self.norm1(sparse.features), inplace=True))
        out = self.conv1(out)
        out = out.replace_feature(F.relu(self.norm2(out.features), inplace=True))
        out = self.conv2(out)
        return out.replace_feature(out.features + identity)


class SpconvSparseUNet(nn.Module):
    """Production sparse 3D U-Net. spconv is imported lazily."""

    def __init__(
        self,
        visual_dim: int,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        visual_channels: int = 256,
        blocks_per_level: int = 2,
        normalization: str = "batch_norm",
    ):
        super().__init__()
        try:
            import spconv.pytorch as spconv
            from spconv.core import ConvAlgo
        except ImportError as exc:
            raise ImportError(
                "The spconv backend requires spconv 2.x. Install a CUDA wheel "
                "matching the runtime, for example `pip install spconv-cu126==2.3.8`."
            ) from exc

        if len(channels) < 2:
            raise ValueError("Sparse U-Net needs at least two resolution levels")
        self.spconv = spconv
        # Keep the algorithm explicit so runtime changes cannot silently alter
        # training numerics. The validated CUDA 12.8 / RTX 4090 environment
        # uses SpConv's performant masked implicit GEMM implementation.
        self.conv_algo = ConvAlgo.MaskImplicitGemm
        self.channels = tuple(int(v) for v in channels)
        self.num_downsamples = len(self.channels) - 1
        self.normalization = str(normalization)
        if self.normalization not in SSR_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported SSR normalization {self.normalization!r}; "
                f"expected one of {SSR_NORMALIZATIONS}"
            )

        self.input_projection = spconv.SubMConv3d(
            3,
            self.channels[0],
            1,
            bias=False,
            indice_key="ssr_input",
            algo=self.conv_algo,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _SparseResidualBlock(
                            spconv,
                            width,
                            f"enc_{level}_{block}",
                            self.conv_algo,
                            self.normalization,
                        )
                        for block in range(blocks_per_level)
                    ]
                )
                for level, width in enumerate(self.channels)
            ]
        )
        self.downsample = nn.ModuleList(
            [
                spconv.SparseConv3d(
                    self.channels[level],
                    self.channels[level + 1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key=f"ssr_down_{level}",
                    algo=self.conv_algo,
                )
                for level in range(self.num_downsamples)
            ]
        )

        self.visual_projection = nn.Linear(visual_dim, visual_channels)
        self.bottleneck_fusion = spconv.SubMConv3d(
            self.channels[-1] + visual_channels,
            self.channels[-1],
            kernel_size=1,
            bias=False,
            indice_key="ssr_visual_fusion",
            algo=self.conv_algo,
        )

        self.upsample = nn.ModuleList(
            [
                spconv.SparseInverseConv3d(
                    self.channels[level + 1],
                    self.channels[level],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"ssr_down_{level}",
                    algo=self.conv_algo,
                )
                for level in range(self.num_downsamples)
            ]
        )
        self.decoder_fusion = nn.ModuleList(
            [
                spconv.SubMConv3d(
                    2 * self.channels[level],
                    self.channels[level],
                    kernel_size=1,
                    bias=False,
                    indice_key=f"ssr_dec_fuse_{level}",
                    algo=self.conv_algo,
                )
                for level in range(self.num_downsamples)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        _SparseResidualBlock(
                            spconv,
                            self.channels[level],
                            f"dec_{level}_{block}",
                            self.conv_algo,
                            self.normalization,
                        )
                        for block in range(blocks_per_level)
                    ]
                )
                for level in range(self.num_downsamples)
            ]
        )
        self.output_layer = spconv.SubMConv3d(
            self.channels[0],
            1,
            kernel_size=1,
            bias=True,
            indice_key="ssr_output",
            algo=self.conv_algo,
        )
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _inject_visual_features(
        self,
        sparse,
        visual_features: torch.Tensor,
        image_size: Tuple[int, int],
    ):
        level_scale = 2**self.num_downsamples
        gathered = sample_visual_features_for_voxels(
            visual_features.float(),
            sparse.indices,
            image_size,
            level_scale,
        )
        gathered = self.visual_projection(gathered)
        sparse = sparse.replace_feature(torch.cat((sparse.features, gathered), dim=-1))
        return self.bottleneck_fusion(sparse)

    def forward(
        self,
        shell: VoxelizedShell,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        sparse = self.spconv.SparseConvTensor(
            features=shell.features,
            indices=shell.coordinates,
            spatial_shape=list(shell.spatial_shape),
            batch_size=shell.batch_size,
        )
        sparse = self.input_projection(sparse)

        skips = []
        active_counts = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                sparse = block(sparse)
            skips.append(sparse)
            active_counts.append(int(sparse.indices.shape[0]))
            if level < self.num_downsamples:
                sparse = self.downsample[level](sparse)

        sparse = self._inject_visual_features(sparse, visual_features, shell.image_size)
        for level in reversed(range(self.num_downsamples)):
            sparse = self.upsample[level](sparse)
            skip = skips[level]
            skip_features = gather_features_at_coordinates(
                skip.features, skip.indices, sparse.indices, sparse.spatial_shape
            )
            sparse = sparse.replace_feature(torch.cat((sparse.features, skip_features), dim=-1))
            sparse = self.decoder_fusion[level](sparse)
            for block in self.decoder_blocks[level]:
                sparse = block(sparse)

        sparse = self.output_layer(sparse)
        residual = gather_features_at_coordinates(
            sparse.features, sparse.indices, shell.coordinates, shell.spatial_shape
        )
        height, width = shell.image_size
        residual = residual[:, 0].reshape(shell.batch_size, height, width)
        return residual, active_counts


class _DenseResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.norm1(values), inplace=True)) * active
        out = self.conv2(F.relu(self.norm2(out), inplace=True)) * active
        return (values + out) * active


class ReferenceSparseUNet(nn.Module):
    """
    Dense masked reference implementation for tiny CPU tests.

    It preserves active-shell masking and the multi-scale data flow, but is not
    intended for full-resolution use or performance comparisons.
    """

    def __init__(
        self,
        visual_dim: int,
        channels: Sequence[int] = (8, 16, 32),
        visual_channels: int = 16,
        blocks_per_level: int = 1,
        max_dense_voxels: int = 2_000_000,
    ):
        super().__init__()
        self.channels = tuple(int(v) for v in channels)
        self.num_downsamples = len(self.channels) - 1
        self.max_dense_voxels = int(max_dense_voxels)
        self.input_projection = nn.Conv3d(3, self.channels[0], 1, bias=False)
        self.encoder_blocks = nn.ModuleList(
            [
                nn.ModuleList([_DenseResidualBlock(c) for _ in range(blocks_per_level)])
                for c in self.channels
            ]
        )
        self.downsample = nn.ModuleList(
            [
                nn.Conv3d(self.channels[i], self.channels[i + 1], 2, stride=2, bias=False)
                for i in range(self.num_downsamples)
            ]
        )
        self.visual_projection = nn.Conv2d(visual_dim, visual_channels, 1)
        self.bottleneck_fusion = nn.Conv3d(
            self.channels[-1] + visual_channels, self.channels[-1], 1, bias=False
        )
        self.upsample = nn.ModuleList(
            [
                nn.ConvTranspose3d(
                    self.channels[i + 1], self.channels[i], 2, stride=2, bias=False
                )
                for i in range(self.num_downsamples)
            ]
        )
        self.decoder_fusion = nn.ModuleList(
            [nn.Conv3d(2 * self.channels[i], self.channels[i], 1, bias=False) for i in range(self.num_downsamples)]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                nn.ModuleList([_DenseResidualBlock(self.channels[i]) for _ in range(blocks_per_level)])
                for i in range(self.num_downsamples)
            ]
        )
        self.output_layer = nn.Conv3d(self.channels[0], 1, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        shell: VoxelizedShell,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        z_size, y_size, x_size = shell.spatial_shape
        dense_voxels = shell.batch_size * z_size * y_size * x_size
        if dense_voxels > self.max_dense_voxels:
            raise RuntimeError(
                f"Reference backend would allocate {dense_voxels:,} dense voxels; "
                f"limit is {self.max_dense_voxels:,}. Use backend='spconv'."
            )

        values = shell.features.new_zeros((shell.batch_size, 3, z_size, y_size, x_size))
        active = shell.features.new_zeros((shell.batch_size, 1, z_size, y_size, x_size))
        batch, depth, row, col = shell.coordinates.to(torch.long).unbind(dim=-1)
        values[batch, :, depth, row, col] = shell.features
        active[batch, :, depth, row, col] = 1
        values = self.input_projection(values) * active

        skips = []
        active_skips = []
        active_counts = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                values = block(values, active)
            skips.append(values)
            active_skips.append(active)
            active_counts.append(int(active.sum().item()))
            if level < self.num_downsamples:
                values = self.downsample[level](values)
                active = F.max_pool3d(active, 2, stride=2)
                values = values * active

        visual = self.visual_projection(
            F.interpolate(
                visual_features.float(),
                size=values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        visual = visual[:, :, None].expand(-1, -1, values.shape[-3], -1, -1)
        values = self.bottleneck_fusion(torch.cat((values, visual), dim=1)) * active

        for level in reversed(range(self.num_downsamples)):
            values = self.upsample[level](values)
            target_shape = skips[level].shape[-3:]
            values = values[..., : target_shape[0], : target_shape[1], : target_shape[2]]
            active = active_skips[level]
            values = self.decoder_fusion[level](torch.cat((values, skips[level]), dim=1)) * active
            for block in self.decoder_blocks[level]:
                values = block(values, active)

        residual_grid = self.output_layer(values)
        residual = residual_grid[batch, 0, depth, row, col]
        height, width = shell.image_size
        return residual.reshape(shell.batch_size, height, width), active_counts


class SelfGuidedSparseRefiner(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        voxel_resolution: float = 200.0,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        visual_channels: int = 256,
        blocks_per_level: int = 2,
        backend: str = "spconv",
        reference_max_dense_voxels: int = 2_000_000,
        normalization: str = "batch_norm",
    ):
        super().__init__()
        self.voxel_resolution = float(voxel_resolution)
        self.backend = backend
        if backend == "spconv":
            self.unet = SpconvSparseUNet(
                visual_dim=visual_dim,
                channels=channels,
                visual_channels=visual_channels,
                blocks_per_level=blocks_per_level,
                normalization=normalization,
            )
        elif backend == "reference":
            self.unet = ReferenceSparseUNet(
                visual_dim=visual_dim,
                channels=channels,
                visual_channels=visual_channels,
                blocks_per_level=blocks_per_level,
                max_dense_voxels=reference_max_dense_voxels,
            )
        else:
            raise ValueError(f"Unsupported sparse backend: {backend!r}")

    @property
    def num_downsamples(self) -> int:
        return self.unet.num_downsamples

    def forward(
        self,
        factorized: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        factorized = factorized.float()
        visual_features = visual_features.float()
        shell = voxelize_factorized(
            factorized,
            voxel_resolution=self.voxel_resolution,
            num_downsamples=self.num_downsamples,
        )
        residual, active_counts = self.unet(shell, visual_features)
        stats: Dict[str, object] = dict(shell.statistics())
        stats["active_voxels_per_level"] = active_counts
        stats["spatial_shape"] = shell.spatial_shape
        return residual, stats


@contextmanager
def stateless_batch_statistics(module: nn.Module):
    """Use current sparse-feature statistics in eval mode without updating buffers."""
    batch_norms = [
        child
        for child in module.modules()
        if isinstance(child, nn.BatchNorm1d)
    ]
    if not batch_norms:
        raise RuntimeError("The selected module contains no BatchNorm1d layers")
    snapshots = [
        (
            child,
            child.running_mean,
            child.running_var,
            child.num_batches_tracked,
            child.track_running_stats,
        )
        for child in batch_norms
    ]
    try:
        for child in batch_norms:
            child.running_mean = None
            child.running_var = None
            child.num_batches_tracked = None
            child.track_running_stats = False
        yield
    finally:
        for (
            child,
            running_mean,
            running_var,
            num_batches_tracked,
            track_running_stats,
        ) in snapshots:
            child.running_mean = running_mean
            child.running_var = running_var
            child.num_batches_tracked = num_batches_tracked
            child.track_running_stats = track_running_stats


def capture_batch_norm_running_state(
    module: nn.Module,
) -> Dict[str, Dict[str, torch.Tensor]]:
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, child in module.named_modules():
        if not isinstance(child, nn.BatchNorm1d):
            continue
        if (
            child.running_mean is None
            or child.running_var is None
            or child.num_batches_tracked is None
        ):
            raise RuntimeError(
                f"BatchNorm running buffers are unavailable for {name}"
            )
        state[name] = {
            "running_mean": child.running_mean.detach().cpu().clone(),
            "running_var": child.running_var.detach().cpu().clone(),
            "num_batches_tracked": (
                child.num_batches_tracked.detach().cpu().clone()
            ),
        }
    if not state:
        raise RuntimeError("The selected module contains no BatchNorm1d layers")
    return state


def load_batch_norm_running_state(
    module: nn.Module,
    state: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    modules = {
        name: child
        for name, child in module.named_modules()
        if isinstance(child, nn.BatchNorm1d)
    }
    if set(modules) != set(state):
        raise RuntimeError(
            "BatchNorm state names do not match the selected module"
        )
    for name, child in modules.items():
        source = state[name]
        if (
            child.running_mean is None
            or child.running_var is None
            or child.num_batches_tracked is None
        ):
            raise RuntimeError(
                f"BatchNorm running buffers are unavailable for {name}"
            )
        child.running_mean.copy_(
            source["running_mean"].to(
                device=child.running_mean.device,
                dtype=child.running_mean.dtype,
            )
        )
        child.running_var.copy_(
            source["running_var"].to(
                device=child.running_var.device,
                dtype=child.running_var.dtype,
            )
        )
        child.num_batches_tracked.copy_(
            source["num_batches_tracked"].to(
                device=child.num_batches_tracked.device,
                dtype=child.num_batches_tracked.dtype,
            )
        )
