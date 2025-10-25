## Coordinate Conventions

| Space | Symbol | Variable Name | Type | Units | Used for |
|-------|--------|---------------|------|-------|-----------|
| Pixel | (u, v) | `px_uv` | Vector2<f32> | pixels | LK, detectors, seeds |
| Camera-normalized ray | $(\hat{x}, \hat{y})$ so that  $(\hat{x}, \hat{y}, 1)$ is bearing | `norm_xy` | Vector2<f64> | unitless, typically ~[-1,1] | Essential matrix, PnP, Triangulation |
| Camera | $C = (x_C, y_C, z_C)$ | `cam_xyz` | Point3<f64> | meters | Triangulated 3D points |
| World | $W = (x_W, y_W, z_W)$ | `world_xyz` | Point3<f64> | meters | Map landmarks |
| Intrinsics | $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ | `intrinsics` | CameraIntrinsics | - | Pixel <-> Normalized |
| Pose | $(R_{wc}, t_{wc})$ pose_wc : Camera -> World (engine, map, trajectory) <br> $(R_{cw}, t_{cw})$ pose_cw : World -> Camera (PnP, reprojection, projection model x = K [R\|t] X_world) | `pose_wc`, `pose_cw` | TransformSE3 | - | See below |

PnP works in pose_cw (world→camera) because that's the standard in epipolar geometry / OpenCV. The engine stores pose_wc (camera→world) for anchors, AR world stability, and user-facing state. At the boundary we invert.