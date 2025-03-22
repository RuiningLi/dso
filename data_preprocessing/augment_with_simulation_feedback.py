from functools import partial
from glob import glob
import multiprocessing
import os
import re

import numpy as np
from tqdm import tqdm

from .simulation_utils import simulate_models


def process_subdir(subdir, root_dir, num_models_per_image, hard_angle_threshold, pattern):
    subdir_path = os.path.join(root_dir, subdir)
    
    # Dictionary to count models per image_id
    image_id_count = {}
    
    # Loop through .glb files in the subdirectory
    for filename in os.listdir(subdir_path):
        if filename.endswith(".glb"):
            match = pattern.match(filename)
            if match:
                image_id = int(match.group(1))  # Extract image_id
                image_id_count[image_id] = image_id_count.get(image_id, 0) + 1
    
    # Find image_ids that have exactly num_models_per_image models
    available_images = [image_id for image_id, count in image_id_count.items() if count == num_models_per_image]
    uniform_images = []

    all_model_paths = sorted(glob(os.path.join(subdir_path, "*.glb")))
    angles = simulate_models(all_model_paths, up_dir="y", num_workers=1)
    if angles is None:
        return None, None, None
    if angles.min() > hard_angle_threshold or angles.max() < hard_angle_threshold:
        pass
    else:
        all_model_image_ids = np.array([int(os.path.splitext(os.path.basename(mp))[0].split("_")[0]) for mp in all_model_paths]).astype(np.int8)
        all_model_glb_ids = np.array([int(os.path.splitext(os.path.basename(mp))[0].split("_")[1]) for mp in all_model_paths]).astype(np.int8)
        angles = angles.astype(np.float16)

        info_dict = {
            "image_ids": all_model_image_ids,
            "glb_ids": all_model_glb_ids,
            "angles": angles,
        }
        np.savez(os.path.join(subdir_path, "model_info.npz"), **info_dict)

    # Save as a NumPy array under the subdirectory
    if available_images:
        npy_path = os.path.join(subdir_path, "available_images.npy")

        for image_id in available_images:
            angle_path = os.path.join(subdir_path, f"{image_id:03d}_angles.npy")
            if not os.path.exists(angle_path):
                glb_paths = sorted(glob(os.path.join(subdir_path, f"{image_id:03d}_*.glb")))
                angles = simulate_models(glb_paths, up_dir="y", num_workers=1)
                if angles is None: 
                    print(f"Failed to simulate models for {subdir} image_id: {image_id}")
                    continue
                np.save(angle_path, angles)
            else:
                angles = np.load(angle_path)
            if angles.min() > hard_angle_threshold or angles.max() < hard_angle_threshold:
                uniform_images.append(image_id)

        print(f"Removing {len(uniform_images)} uniform images from {subdir}")
        available_images = [img_id for img_id in available_images if img_id not in uniform_images]
        if len(available_images) > 0:
            np.save(npy_path, np.array(available_images, dtype=int))
            print(f"Saved {len(available_images)} image_ids in: {npy_path}")
        else:
            print(f"No available images for {subdir}")
            
    return len(available_images), 1 if available_images else 0, False


def compute_and_save_simulation_feedback(root_dir, num_models_per_image=16, stable_threshold=20):
    # Regex pattern to extract image_id and model_id
    pattern = re.compile(r"(\d{3})_\d{3}\.glb")

    subdir_with_available_images = set()

    subdirs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
    process_fn = partial(process_subdir, root_dir=root_dir, num_models_per_image=num_models_per_image, 
                        hard_angle_threshold=stable_threshold, pattern=pattern)

    total_available_images = 0
    subdir_with_available_images = 0
    nonuniform_objects = 0
    simulation_failed = 0

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_fn, subdirs), total=len(subdirs)))
        
        for num_available_images, has_available, is_uniform in results:
            if num_available_images is None and has_available is None and is_uniform is None:
                simulation_failed += 1
            else:
                total_available_images += num_available_images
                subdir_with_available_images += has_available
                if not is_uniform:
                    nonuniform_objects += 1
                    
    print(f"Total available images: {total_available_images}")
    print(f"Total objects with available images: {subdir_with_available_images}")
    print(f"Total non-uniform objects: {nonuniform_objects}")
    print(f"Simulation failed: {simulation_failed}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, required=True)
    parser.add_argument("--num_models_per_image", type=int, required=True)
    parser.add_argument("--stable_threshold", type=int, default=20)
    args = parser.parse_args()
    compute_and_save_simulation_feedback(args.root_directory, num_models_per_image=args.num_models_per_image, stable_threshold=args.stable_threshold)
