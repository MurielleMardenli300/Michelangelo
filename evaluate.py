from train import ShapeVAEModule
import torch, open3d as o3d, numpy as np, trimesh

model = ShapeVAEModule.load_from_checkpoint(
    '/home/mardenlim/codebases/Michelangelo/output/abdomen_shapevae/checkpoints/epoch=047-val/loss=0.3144.ckpt'
)
model.eval().cuda()

pcd = o3d.io.read_point_cloud(
    '/volatile/Datasets/Varian_Motion/point_clouds/norm_pc/test1/pointcloud_00001.ply'
)
pts = np.asarray(pcd.points, np.float32)
nrm = np.asarray(pcd.normals, np.float32)

# ── FIX: this MUST match AbdominalDataset._load_pointcloud exactly ─────────
# Your original script had this commented out, which meant the model was
# receiving raw-scale, uncentered point coordinates -- completely outside
# the [-0.9995, 0.9995] distribution it was trained on. That mismatch is the
# most likely reason reconstruct() was returning None.
centroid = pts.mean(0)
pts = pts - centroid
scale = np.abs(pts).max()
pts = pts / (scale + 1e-8)
pts = np.clip(pts * 0.9995, -0.9995, 0.9995)
nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)
# ─────────────────────────────────────────────────────────────────────────

print(f"[debug] pts range after normalization: [{pts.min():.4f}, {pts.max():.4f}] "
      f"(expect close to [-0.9995, 0.9995])")

surface = torch.from_numpy(np.concatenate([pts, nrm], -1)).float().unsqueeze(0).cuda()

outputs = model.reconstruct(surface, octree_depth=7)
if outputs[0] is not None:
    print("Outputs not none!")
    mesh = trimesh.Trimesh(outputs[0].mesh_v, outputs[0].mesh_f)
    mesh.export('reconstruction_test.obj')
    print(f"Saved reconstruction_test.obj: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
else:
    print("Still None -- see [reconstruct] diagnostic prints above for xyz range "
          "and has_surface status.")