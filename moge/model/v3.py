from __future__ import annotations

from numbers import Number
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift
from .ssr import (
    SelfGuidedSparseRefiner,
    factorize_points,
    load_batch_norm_running_state,
    smooth_bound_log_depth_residual,
)
from .v2 import MoGeModel as MoGeModelV2


class MoGeModel(MoGeModelV2):
    """
    Paper-faithful MoGe-3 reconstruction built on the released MoGe-2 model.

    The official MoGe-3 architecture and weights are not available yet. Sparse
    U-Net widths and block counts are therefore explicit, checkpointed config.
    """

    def __init__(
        self,
        *args,
        ssr: Optional[Dict[str, Any]] = None,
        num_refinement_steps: int = 3,
        predict_normal: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.predict_normal = bool(predict_normal)
        if not self.predict_normal and hasattr(self, "normal_head"):
            del self.normal_head
        encoder_config = kwargs.get("encoder")
        if encoder_config is None and args:
            encoder_config = args[0]
        if encoder_config is None:
            raise ValueError("MoGe-3 requires the base encoder configuration")

        ssr_config = {
            "visual_dim": int(encoder_config["dim_out"]),
            "voxel_resolution": 200.0,
            "channels": [32, 64, 128, 256, 512],
            "visual_channels": 256,
            "blocks_per_level": 2,
            "backend": "spconv",
            "normalization": "batch_norm",
        }
        if ssr is not None:
            ssr_config.update(ssr)
        self.ssr_config = ssr_config
        self.ssr = SelfGuidedSparseRefiner(**ssr_config)
        self.default_num_refinement_steps = int(num_refinement_steps)

    def enable_gradient_checkpointing(self):
        # Sparse layers are not wrapped by the MoGe-2 checkpoint helper.
        super().enable_gradient_checkpointing()

    def _forward_base(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device = image.device
        geometry_dtype = torch.float32

        aspect_ratio = img_w / img_h
        base_h = (num_tokens / aspect_ratio) ** 0.5
        base_w = (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        use_dino_bfloat16 = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_dino_bfloat16,
        ):
            visual_features, cls_token = self.encoder(
                image, base_h, base_w, return_class_token=True
            )

        # The paper keeps DINO in bfloat16 and all geometry modules in fp32.
        with torch.autocast(device_type=device.type, enabled=False):
            visual_features = visual_features.float()
            cls_token = cls_token.float()
            features: List[Optional[torch.Tensor]] = [
                visual_features,
                None,
                None,
                None,
                None,
            ]
            for level in range(5):
                uv = normalized_view_plane_uv(
                    width=base_w * 2**level,
                    height=base_h * 2**level,
                    aspect_ratio=aspect_ratio,
                    dtype=geometry_dtype,
                    device=device,
                )
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
                if features[level] is None:
                    features[level] = uv
                else:
                    features[level] = torch.cat((features[level], uv), dim=1)

            decoded_features = self.neck(features)
            points, normal, mask = (
                getattr(self, head)(decoded_features)[-1] if hasattr(self, head) else None
                for head in ["points_head", "normal_head", "mask_head"]
            )
            metric_scale = self.scale_head(cls_token) if hasattr(self, "scale_head") else None
        points, normal, mask = (
            F.interpolate(value, (img_h, img_w), mode="bilinear", align_corners=False)
            if value is not None
            else None
            for value in [points, normal, mask]
        )

        if points is not None:
            points = self._remap_points(points.permute(0, 2, 3, 1))
        if normal is not None:
            normal = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        output = {
            "points": points,
            "normal": normal,
            "mask": mask,
            "metric_scale": metric_scale,
            "_visual_features": visual_features,
        }
        return {key: value for key, value in output.items() if value is not None}

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor] = 2500,
        num_refinement_steps: Optional[int] = None,
        return_intermediates: bool = False,
        detach_base_from_refiner: bool = False,
        smooth_log_depth_residual_bound: Optional[float] = None,
        max_voxel_depth_span: Optional[int] = None,
        ssr_batch_norm_states: Optional[
            List[Dict[str, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> Dict[str, Any]:
        if num_refinement_steps is None:
            num_refinement_steps = self.default_num_refinement_steps
        if not 0 <= int(num_refinement_steps) <= 7:
            raise ValueError("num_refinement_steps must be in [0, 7]")
        if (
            ssr_batch_norm_states is not None
            and len(ssr_batch_norm_states) < int(num_refinement_steps)
        ):
            raise ValueError(
                "SSR BatchNorm states must cover every refinement iteration"
            )

        output = self._forward_base(image, num_tokens)
        base_points = output["points"]
        visual_features = output.pop("_visual_features")
        points_sequence = [base_points]
        residuals = []
        raw_residuals = []
        voxel_stats = []

        if num_refinement_steps:
            with torch.autocast(device_type=image.device.type, enabled=False):
                factorized = factorize_points(base_points.float())
                visual_for_refiner = visual_features.float()
                refined_points = base_points.float()
                if detach_base_from_refiner:
                    factorized = factorized.detach()
                    visual_for_refiner = visual_for_refiner.detach()
                    refined_points = refined_points.detach()

                for iteration in range(int(num_refinement_steps)):
                    if ssr_batch_norm_states is not None:
                        load_batch_norm_running_state(
                            self.ssr,
                            ssr_batch_norm_states[iteration],
                        )
                    raw_residual, stats = self.ssr(
                        factorized,
                        visual_for_refiner,
                        max_depth_span=max_voxel_depth_span,
                    )
                    residual = smooth_bound_log_depth_residual(
                        raw_residual,
                        smooth_log_depth_residual_bound,
                    )
                    factorized = torch.cat(
                        (
                            factorized[..., :2],
                            factorized[..., 2:3] + residual[..., None],
                        ),
                        dim=-1,
                    )
                    # u,v stay fixed, so a log-depth update scales X,Y,Z
                    # equally. Multiplication by exp(0)=1 makes the
                    # zero-initialized SSR a bitwise identity.
                    refined_points = refined_points * residual.exp()[..., None]
                    raw_residuals.append(raw_residual)
                    residuals.append(residual)
                    voxel_stats.append(stats)
                    points_sequence.append(refined_points)
            output["points"] = points_sequence[-1]

        if return_intermediates:
            output["points_sequence"] = points_sequence
            output["log_depth_residuals"] = residuals
            output["raw_log_depth_residuals"] = raw_residuals
            output["voxel_stats"] = voxel_stats
        return output

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = True,
        num_refinement_steps: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        height, width = image.shape[-2:]
        aspect_ratio = width / height
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=use_fp16 and self.dtype != torch.float16,
        ):
            output = self.forward(
                image,
                num_tokens=num_tokens,
                num_refinement_steps=num_refinement_steps,
                return_intermediates=return_intermediates,
            )
        points, normal, mask, metric_scale = (
            output.get(key) for key in ["points", "normal", "mask", "metric_scale"]
        )
        points, normal, mask, metric_scale, fov_x = (
            value.float() if isinstance(value, torch.Tensor) else value
            for value in [points, normal, mask, metric_scale, fov_x]
        )

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            mask_binary = mask > 0.5 if mask is not None else None
            if points is not None:
                if fov_x is None:
                    focal, shift = recover_focal_shift(points, mask_binary)
                else:
                    focal = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
                    focal = focal / torch.tan(
                        torch.deg2rad(
                            torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2
                        )
                    )
                    if focal.ndim == 0:
                        focal = focal[None].expand(points.shape[0])
                    _, shift = recover_focal_shift(points, mask_binary, focal=focal)
                fx = focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
                fy = focal / 2 * (1 + aspect_ratio**2) ** 0.5
                intrinsics = utils3d.pt.intrinsics_from_focal_center(
                    fx,
                    fy,
                    torch.tensor(0.5, device=points.device, dtype=points.dtype),
                    torch.tensor(0.5, device=points.device, dtype=points.dtype),
                )
                points[..., 2] += shift[..., None, None]
                if mask_binary is not None:
                    mask_binary &= points[..., 2] > 0
                depth = points[..., 2].clone()
            else:
                depth, intrinsics = None, None

            if force_projection and depth is not None:
                points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics)
            if metric_scale is not None:
                if points is not None:
                    points *= metric_scale[:, None, None, None]
                if depth is not None:
                    depth *= metric_scale[:, None, None]
            if apply_mask and mask_binary is not None:
                if points is not None:
                    points = torch.where(mask_binary[..., None], points, torch.inf)
                if depth is not None:
                    depth = torch.where(mask_binary, depth, torch.inf)
                if normal is not None:
                    normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

        result: Dict[str, Any] = {
            "points": points,
            "intrinsics": intrinsics,
            "depth": depth,
            "mask": mask_binary,
            "normal": normal,
        }
        if return_intermediates:
            result["points_sequence"] = output["points_sequence"]
            result["log_depth_residuals"] = output["log_depth_residuals"]
            result["voxel_stats"] = output["voxel_stats"]
        result = {key: value for key, value in result.items() if value is not None}
        if omit_batch_dim:
            result = {
                key: value.squeeze(0) if isinstance(value, torch.Tensor) else value
                for key, value in result.items()
            }
        return result
