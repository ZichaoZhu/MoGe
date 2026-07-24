# MoGe-3 training-data inventory

The paper uses 17 synthetic datasets and 8 real/sensor-reconstructed datasets.
SSR receives gradients only from the synthetic group. Sizes marked `audit`
must be verified against the exact release and selected modalities before any
download; dataset names alone are not precise enough to quote a safe byte
count.

| Dataset | Group | Required for base | Required for SSR | Server status | Size / acquisition audit |
|---|---|---|---|---|---|
| 3D Ken Burns | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit project release |
| ApolloSynthetic | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit Apollo terms and release |
| EDEN | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit official release |
| GTA-SfM | synthetic | RGB, depth, camera | pixel-exact geometry | absent | restricted acquisition; audit |
| Hypersim | synthetic | RGB, depth, intrinsics | pixel-exact geometry | absent; unusable RGB+normal derivative removed | full release is roughly hundreds of GB; select RGB/depth/camera only |
| IRS | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit official release |
| MatrixCity | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit exact split/modalities |
| MidAir | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit exact trajectories/modalities |
| MVS-Synth | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit official release |
| Objaverse | synthetic | rendered RGB/depth/camera | pixel-exact geometry | absent | rendering pipeline and object subset unspecified |
| OmniWorld | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit paper release/access |
| Structured3D | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit 2D render subset |
| Synscapes | synthetic | RGB, depth, camera | pixel-exact geometry | absent | licensed download; audit |
| Synthia | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit sequences/modalities |
| TartanAir | synthetic | RGB, depth, camera | pixel-exact geometry | absent | multi-terabyte full release; choose environments |
| UnrealStereo4K | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit official release |
| UrbanSyn | synthetic | RGB, depth, camera | pixel-exact geometry | absent | audit official release |
| A2D2 | real/sensor | RGB, depth, intrinsics | no | absent | audit camera/LiDAR subset |
| ARKitScenes | real/sensor | RGB, depth, intrinsics | no | absent | audit raw vs processed split |
| Argoverse2 | real/sensor | RGB, depth, intrinsics | no | absent | audit sensor split |
| BlendedMVS | reconstructed | RGB, depth, camera | no | absent | audit low/high-resolution release |
| MegaDepth | reconstructed | RGB, depth, camera | no | absent | audit SfM/MVS preprocessing |
| ScanNet++ | real/sensor | RGB, depth, intrinsics | no | absent | application/terms required |
| Taskonomy | real/sensor | RGB, depth, camera | no | RGB/depth/mask candidate present | retained approximately 469 GB; pairing/camera audit pending |
| Waymo | real/sensor | RGB, depth, intrinsics | no | absent | account/terms and conversion audit |

## Existing safe-root assets

- The former `hypersim_dpt_52k_trainval_camera_normal` derivative (about 51 GB)
  had 53,655 RGB images and 53,655 camera-normal files but no paired depth or
  camera calibration. It and its empty held-out companion were removed.
- `omnidata_3dcc_original`: retained Taskonomy-like RGB, Euclidean depth, and
  valid masks (469 GB total): 1,235,983 RGB files, 1,253,911 Euclidean-depth
  files, and 1,250,407 valid-mask files. The normal and reshading trees were
  removed because this reproduction does not predict normals.
- Empty `taskonomy_100gb_3dcc` and `taskonomy_medium_3dcc` shells were removed.
- no content referenced through `/nas1` is in scope or counted.

No downloader or data converter is enabled in this milestone.
