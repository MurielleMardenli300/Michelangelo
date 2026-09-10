"""
make_watertight_poisson.py

Replaces the fragile manual extrude-and-stitch shell approach with Poisson
surface reconstruction, which reliably produces a genuinely watertight
closed mesh from an oriented point sample -- regardless of how irregular
or multi-component the original open surface's boundary is (the manual
edge-stitching approach broke on real scan data, confirmed by
diagnose_pipeline.py showing is_watertight=False even after "shell"
conversion).

Approach:
  1. Sample a dense point cloud + normals directly off the open surface mesh.
  2. Run Poisson reconstruction (Open3D) -> produces a genuinely closed,
     watertight mesh by construction.
  3. Crop the result back down close to the original surface's bounding
     region (Poisson can extrapolate a "cap" that balloons outward --
     cropping keeps the result faithful to your actual abdomen shape
     rather than an arbitrarily inflated blob).
  4. Verify watertightness and report.

Usage:
    python make_watertight_poisson.py \
        --mesh_dir /volatile/Datasets/Varian_Motion/meshes \
        --out_dir  /volatile/Datasets/Varian_Motion/meshes_watertight \
        --poisson_depth 9
"""

import argparse
import glob
import os
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def cap_boundary_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Guarantees watertight closure by explicitly capping any remaining open
    boundary loop(s) with a fan triangulation from each loop's centroid.

    This is needed because Poisson reconstruction, despite being watertight
    "by construction" in the ideal case, can leave a genuine unclosed hole
    on real open-surface input (confirmed empirically: increasing
    poisson_depth from 9 to 11 did not close it -- this is a structural
    gap at the extrapolated cap boundary, not a resolution issue). This
    function traces the boundary edge loop(s) and closes them directly,
    which is a standard, reliable hole-filling technique.
    """
    mesh = mesh.copy()
    edges = mesh.edges_sorted
    edge_counts = Counter(map(tuple, edges))
    open_edges = [e for e, c in edge_counts.items() if c == 1]
    if not open_edges:
        return mesh  # already watertight, nothing to cap

    adj = defaultdict(list)
    for a, b in open_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited_edges = set()
    loops = []
    for start_edge in open_edges:
        if start_edge in visited_edges or (start_edge[1], start_edge[0]) in visited_edges:
            continue
        loop = [start_edge[0]]
        current = start_edge[1]
        prev = start_edge[0]
        visited_edges.add(start_edge)
        while current != loop[0]:
            loop.append(current)
            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                break  # incomplete/broken loop trace -- skip rather than corrupt geometry
            nxt = neighbors[0]
            visited_edges.add((current, nxt))
            visited_edges.add((nxt, current))
            prev, current = current, nxt
        loops.append(loop)

    new_verts = list(mesh.vertices)
    new_faces = list(mesh.faces)
    for loop in loops:
        if len(loop) < 3:
            continue
        loop_pts = mesh.vertices[loop]
        centroid = loop_pts.mean(axis=0)
        centroid_idx = len(new_verts)
        new_verts.append(centroid)
        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i + 1) % len(loop)]
            new_faces.append([a, b, centroid_idx])

    capped = trimesh.Trimesh(vertices=np.array(new_verts), faces=np.array(new_faces), process=True)
    capped.merge_vertices()
    capped.remove_unreferenced_vertices()
    capped.fix_normals()
    return capped


def close_mesh_poisson(mesh: trimesh.Trimesh, poisson_depth: int = 9,
                        n_sample_points: int = 20000,
                        crop_margin_fraction: float = 0.05) -> trimesh.Trimesh:
    """
    Converts an open surface mesh into a watertight closed mesh via Poisson
    reconstruction.

    IMPORTANT: does NOT crop or density-trim the Poisson output. Both of
    those operations delete faces/vertices without capping the resulting
    cut -- which re-opens the mesh exactly at the cut boundary and destroys
    the watertight property Poisson just gave us. (This was the actual bug
    in the previous version of this function: it reported is_watertight
    False because of its OWN cropping step, not because Poisson failed.)

    Poisson's raw output is watertight by construction. It may include an
    extrapolated "cap" ballooning slightly beyond the real open surface's
    original extent -- that's an accepted tradeoff for correctness here;
    if you need a tighter fit later, use a proper watertight-preserving
    boolean intersection (e.g. trimesh.boolean.intersection with a box,
    which requires a working boolean backend) rather than naive deletion.
    """
    # 1. Sample a dense oriented point cloud off the open surface
    sampled_pts, face_idx = trimesh.sample.sample_surface(mesh, n_sample_points)
    sampled_normals = mesh.face_normals[face_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_pts)
    pcd.normals = o3d.utility.Vector3dVector(sampled_normals)

    # 2. Poisson reconstruction -- produces a genuinely watertight mesh
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False
    )

    verts = np.asarray(poisson_mesh.vertices)
    faces = np.asarray(poisson_mesh.triangles)
    closed = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Only safe, topology-preserving cleanup below -- these don't delete
    # arbitrary geometry, just normalize duplicate/degenerate bookkeeping.
    closed.merge_vertices()
    closed.remove_unreferenced_vertices()
    closed.fix_normals()

    # Poisson can still leave a genuine open hole even at high depth
    # (confirmed empirically -- not a resolution issue). Explicitly cap
    # any remaining boundary loop(s) to guarantee true watertightness.
    if not closed.is_watertight:
        closed = cap_boundary_holes(closed)

    return closed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--poisson_depth", type=int, default=9,
                         help="Higher = more detail but slower and more prone to "
                              "noise-fitting. 8-10 is typical for this kind of surface.")
    parser.add_argument("--n_sample_points", type=int, default=20000)
    parser.add_argument("--crop_margin_fraction", type=float, default=0.05)
    args = parser.parse_args()

    mesh_paths = sorted(glob.glob(os.path.join(args.mesh_dir, "**", "*.stl"), recursive=True))
    if not mesh_paths:
        print(f"No .stl files found under {args.mesh_dir}")
        return

    print(f"Found {len(mesh_paths)} meshes. Closing via Poisson reconstruction "
          f"(depth={args.poisson_depth})...\n")

    n_watertight = 0
    n_failed = 0

    for path in mesh_paths:
        rel_path = os.path.relpath(path, args.mesh_dir)
        out_path = Path(args.out_dir) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            mesh = trimesh.load(path, process=False, force="mesh")

            if mesh.is_watertight:
                mesh.export(str(out_path))
                n_watertight += 1
                print(f"  [already watertight] {rel_path}")
                continue

            closed = close_mesh_poisson(
                mesh,
                poisson_depth=args.poisson_depth,
                n_sample_points=args.n_sample_points,
                crop_margin_fraction=args.crop_margin_fraction,
            )

            status = "watertight" if closed.is_watertight else "STILL NOT WATERTIGHT"
            print(f"  [{status}] {rel_path}: {len(mesh.faces)} -> {len(closed.faces)} faces")

            if closed.is_watertight:
                n_watertight += 1
            else:
                n_failed += 1
                print(f"    [WARN] Poisson reconstruction did not fully close this mesh. "
                      f"Try a higher --poisson_depth, or inspect this file manually.")

            closed.export(str(out_path))

        except Exception as e:
            n_failed += 1
            print(f"  [FAILED] {rel_path}: {e}")

    print(f"\nDone. {n_watertight} watertight, {n_failed} failed/not-watertight, "
          f"out of {len(mesh_paths)} total.")
    print(f"Output saved to {args.out_dir}")


if __name__ == "__main__":
    main()