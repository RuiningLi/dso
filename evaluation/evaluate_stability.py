from contextlib import contextmanager
from functools import partial
import multiprocessing
import os
import signal
from typing import List, Union

import mujoco
import numpy as np
from tqdm import tqdm
import trimesh


def quaternion_to_axis_angle(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    angle = 2 * np.arccos(w)
    sin_theta = np.sqrt(1 - w ** 2)  # sin(theta/2)
    
    zero_mask = sin_theta < 1e-6
    axis = np.zeros(q.shape[:-1] + (3,))
    
    axis = np.where(zero_mask[..., None], 
                    np.array([1., 0., 0.]), 
                    np.stack([x, y, z], axis=-1) / np.where(zero_mask[..., None], 1, sin_theta[..., None]))
    return axis, angle


def get_mujoco_model_and_data(
    v_pos: Union[np.ndarray, List[np.ndarray]], 
    faces: Union[np.ndarray, List[np.ndarray]],
):
    if not isinstance(v_pos, list):
        v_pos = [v_pos]
    if not isinstance(faces, list):
        faces = [faces]

    if len(v_pos) != len(faces):
        assert len(v_pos) == 1 or len(faces) == 1
        if len(v_pos) == 1:
            v_pos = v_pos * len(faces)
        else:
            faces = faces * len(v_pos)
    assert len(v_pos) == len(faces)

    asset_xml, body_xml = "", ""
    for i, (vp, f) in enumerate(zip(v_pos, faces)):
        faces_str = "  ".join(f"{v1:d} {v2:d} {v3:d}" for v1, v2, v3 in f)
        vertices_str = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vp)
        asset_xml += f"""
            <mesh name="mesh_{i}" scale="1 1 1" vertex="{vertices_str}" face="{faces_str}"/>
        """
        body_xml += f"""
            <body name="mesh_body_{i}" pos="0 {i * 5} 0">
                <joint name="free_joint_{i}" type="free" />
                <geom name="mesh_geom_{i}" type="mesh" mesh="mesh_{i}"/>
            </body>
        """

    model_xml = f"""
        <mujoco model="mesh_simulation">
            <asset>
                {asset_xml}
            </asset>

            <worldbody>
                <!-- Ground plane -->
                <geom name="ground_plane" type="plane" pos="0 0 0" size="0 0 1" />
                
                <!-- Mesh object with free joint --> 
                {body_xml}
            </worldbody>
        </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    return model, data


def get_sim_angles(
    v_pos: Union[np.ndarray, List[np.ndarray]],
    faces: Union[np.ndarray, List[np.ndarray]],
    timeout: float = 5.0
):

    def simulate():
        # Convert vertices and faces to a format suitable for MuJoCo
        model, data = get_mujoco_model_and_data(v_pos, faces)
        duration = 10.0  # seconds
        model.opt.timestep = 0.001

        while data.time < duration:
            mujoco.mj_step(model, data)

        rotation = [data.qpos[3:]]
        _, angles = quaternion_to_axis_angle(np.array(rotation))
        angles = np.rad2deg(angles)
        return angles[0]

    try:
        @contextmanager
        def time_limit(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError("Simulation timed out")
            
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(int(timeout))
            
            try:
                yield
            finally:
                signal.alarm(0)
        
        with time_limit(timeout):
            return simulate()
        
    except TimeoutError:
        print("Simulation timed out")
        return None
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        return None


def process_obj_file(mesh_path, up_dir):
    if not os.path.exists(mesh_path):
        return None
    
    mesh = trimesh.util.concatenate(trimesh.load(mesh_path))

    if isinstance(mesh, trimesh.points.PointCloud):
        return None
    
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()

    # Scale vertices to fit in [-0.5, 0.5] cube while preserving aspect ratio
    center = np.mean(mesh.vertices, axis=0)
    mesh.vertices -= center
    max_dim = np.max(np.abs(mesh.vertices))
    if max_dim > 0:  # Avoid division by zero
        scale_factor = 0.5 / max_dim
        mesh.vertices *= scale_factor
    vertices = np.array(mesh.vertices)
    if up_dir == "y":
        vertices[:, [1, 2]] = vertices[:, [2, 1]]
    elif up_dir == "x":
        vertices[:, [0, 2]] = vertices[:, [2, 0]]

    # Make the lowest point to have z = 0
    vertices[:, 2] -= np.min(vertices[:, 2])

    faces = np.array(mesh.faces).astype(np.int32)

    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return None

    return get_sim_angles(vertices, faces)


def process_obj_files(mesh_paths, up_dir, num_workers=None):
    process_fn = partial(process_obj_file, up_dir=up_dir)
    with multiprocessing.Pool(processes=num_workers or multiprocessing.cpu_count()) as pool:
        angles = list(tqdm(pool.imap(process_fn, mesh_paths), total=len(mesh_paths)))
    return np.array(angles)


if __name__ == "__main__":
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_paths", type=str, required=True)
    parser.add_argument("--up_dir", type=str, default="y", choices=["x", "y", "z"])
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    mesh_paths = sorted(glob(args.mesh_paths))

    angles = process_obj_files(mesh_paths, args.up_dir, args.num_workers)
    print("Successfully processed ", len(angles), " meshes")
    angles = np.array(angles)
    print("Average Rotation Angle at Equilibrium: ", angles.mean())
    print("Percentage Stable: ", (angles > 20).mean())
