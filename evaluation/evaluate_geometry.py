import os.path as osp

from glob import glob
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
import trimesh


def icp_align(source, target, max_iterations=50, tolerance=1e-6):
    source_pc = o3d.geometry.PointCloud()
    target_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source)
    target_pc.points = o3d.utility.Vector3dVector(target)

    # Apply ICP
    threshold = 0.02
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, relative_fitness=tolerance)
    )

    # Apply the transformation
    aligned_source = np.asarray(source_pc.transform(icp_result.transformation).points)
    return aligned_source


def compute_metrics(gen_model, gt_model, n_points=10000, tau=0.05):
    gen_points, _ = trimesh.sample.sample_surface(gen_model, n_points)
    gt_points, _ = trimesh.sample.sample_surface(gt_model, n_points)

    gen_points = icp_align(gen_points, gt_points)

    gt_tree = KDTree(gt_points)
    pred_tree = KDTree(gen_points)

    pred_to_gt_distances, _ = gt_tree.query(gen_points)
    gt_to_pred_distances, _ = pred_tree.query(gt_points)

    precision = np.mean(pred_to_gt_distances < tau)
    recall = np.mean(gt_to_pred_distances < tau)

    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * (precision * recall) / (precision + recall)
    chamfer_distance = (np.mean(pred_to_gt_distances) + np.mean(gt_to_pred_distances)) / 2
    return fscore, chamfer_distance


def process_single_mesh(mesh_path, up_dir):
    gen_model = trimesh.util.concatenate(trimesh.load(mesh_path))
    if up_dir == "z":
        gen_model.vertices[:, [1, 2]] = gen_model.vertices[:, [2, 1]]
    elif up_dir != "y":
        raise ValueError(f"Invalid up_dir: {up_dir}")

    object_id = osp.basename(mesh_path).split("-")[0]
    ground_truth_path = glob(osp.join(args.ground_truth_paths, "*", f"{object_id}.glb"))
    assert len(ground_truth_path) == 1, f"Ground truth path {ground_truth_path} does not exist"
    ground_truth_path = ground_truth_path[0]

    gt_model = trimesh.util.concatenate(trimesh.load(ground_truth_path))

    # Scale and center both models to [-0.5, 0.5] ^ 3
    for model in [gen_model, gt_model]:
        bounds = model.bounds
        extents = bounds[1] - bounds[0]
        scale = 1.0 / np.max(extents)
        center = (bounds[1] + bounds[0]) / 2
        translation = np.eye(4)
        translation[:3,3] = -center
        scaling = np.eye(4)
        scaling[:3,:3] *= scale
        model.apply_transform(translation)
        model.apply_transform(scaling)

    fscore, chamfer_distance = compute_metrics(gen_model, gt_model)
    return fscore, chamfer_distance


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_paths", type=str, required=True)
    parser.add_argument("--ground_truth_paths", type=str, required=True)  # This should point to the directory which contains the downloaded objaverse models (.glb files), e.g. /mnt/disks/data/objaverse/hf-objaverse-v1/glbs
    parser.add_argument("--up_dir", type=str, default="y", choices=["x", "y", "z"])
    args = parser.parse_args()

    mesh_paths = sorted(glob(args.mesh_paths))

    results = []
    for mesh_path in tqdm(mesh_paths, total=len(mesh_paths)):
        results.append(process_single_mesh(mesh_path, args.up_dir))

    total_fscore = 0
    total_chamfer_distance = 0
    total_count = len(results)

    all_fscores = []
    all_chamfer_distances = []
    for fscore, chamfer_distance in results:
        if fscore is not None and chamfer_distance is not None:
            total_fscore += fscore
            total_chamfer_distance += chamfer_distance
            all_fscores.append(fscore)
            all_chamfer_distances.append(chamfer_distance)
    
    print(f"Average F-Score: {np.mean(all_fscores)}")
    print(f"Average Chamfer Distance: {np.mean(all_chamfer_distances)}")
