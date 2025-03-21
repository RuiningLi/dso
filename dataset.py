import os
from os import path as osp
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(
        self, 
        root_dir: str = "./data",
        category: str = "objaverse-renderings",
        prompts: Optional[list[str]] = None,
        num_images_per_prompt: int = 10,
        image_ids: Optional[list[int]] = None,
        num_models_per_image: int = 16,
        sample_from_all_multiviews: bool = False,
        stable_threshold: float = 20.0,
    ):
        self.dataset_root = osp.join(root_dir, category)
        self.prompts = prompts if prompts is not None else [d for d in os.listdir(self.dataset_root) if os.path.exists(osp.join(self.dataset_root, d, "available_images.npy"))]
        print(f"Number of prompts: {len(self.prompts)}")
        self.image_ids = image_ids if image_ids is not None else list(range(num_images_per_prompt))
        self.num_models_per_image = num_models_per_image
        self.sample_from_all_multiviews = sample_from_all_multiviews
        self.stable_threshold = stable_threshold

    def __getitem__(self, index):
        if self.sample_from_all_multiviews:
            while True:
                try:
                    while True:
                        prompt = random.choice(self.prompts)
                        prompt_root = osp.join(self.dataset_root, prompt)
                        if not osp.exists(osp.join(prompt_root, "model_info.npz")):
                            continue
                        model_info = np.load(osp.join(prompt_root, "model_info.npz"))
                        image_ids = model_info["image_ids"]
                        glb_ids = model_info["glb_ids"]
                        angles = model_info["angles"]
                        assert len(image_ids) == len(glb_ids) == len(angles)
                        if not np.logical_and(angles[:, 0] < self.stable_threshold, glb_ids < self.num_models_per_image).any():
                            continue
                        if not np.logical_and(angles[:, 0] >= self.stable_threshold, glb_ids < self.num_models_per_image).any():
                            continue
                        break

                    image_id = random.choice(np.unique(image_ids))
                    cond_path = osp.join(prompt_root, f"{image_id:03d}_cond.pt")
                    cond = torch.load(cond_path, weights_only=True, map_location="cpu")

                    indices = np.arange(len(image_ids))
                    good_indices = indices[np.logical_and(angles[:, 0] < self.stable_threshold, glb_ids < self.num_models_per_image)]
                    bad_indices = indices[np.logical_and(angles[:, 0] >= self.stable_threshold, glb_ids < self.num_models_per_image)]
                    assert good_indices.size > 0 and bad_indices.size > 0
                    good_index = np.random.choice(good_indices)
                    bad_index = np.random.choice(bad_indices)
                    image_id_win = image_ids[good_index]
                    image_id_loss = image_ids[bad_index]
                    model_win = glb_ids[good_index]
                    model_loss = glb_ids[bad_index]
                    
                    model_win_sparse_x0_path = osp.join(prompt_root, f"{image_id_win:03d}_sparse_sample_{model_win:03d}.pt")
                    model_loss_sparse_x0_path = osp.join(prompt_root, f"{image_id_loss:03d}_sparse_sample_{model_loss:03d}.pt")

                    model_win_sparse_x0 = torch.load(model_win_sparse_x0_path, weights_only=True, map_location="cpu")
                    model_loss_sparse_x0 = torch.load(model_loss_sparse_x0_path, weights_only=True, map_location="cpu")

                    return {
                        "prompt": prompt,
                        "cond": cond,
                        "model_win_sparse_x0": model_win_sparse_x0,
                        "model_loss_sparse_x0": model_loss_sparse_x0,
                    }

                except FileNotFoundError as e:
                    pass
                except Exception as e:
                    print(e)

        else:
            while True:
                prompt = random.choice(self.prompts)
                prompt_root = osp.join(self.dataset_root, prompt)
                if not osp.exists(osp.join(prompt_root, "available_images.npy")):
                    continue
                available_images = np.load(osp.join(prompt_root, "available_images.npy"))
                image_id = random.choice(available_images)
                break
            
            cond_path = osp.join(prompt_root, f"{image_id:03d}_cond.pt")
            cond = torch.load(cond_path, weights_only=True, map_location="cpu")

            angle_path = osp.join(prompt_root, f"{image_id:03d}_angles.npy")
            angles = np.load(angle_path)
            good_model_ids = np.arange(self.num_models_per_image)
            winning_candidate_ids = good_model_ids[angles[good_model_ids, 0] < self.stable_threshold]
            loss_candidate_ids = good_model_ids[angles[good_model_ids, 0] >= self.stable_threshold]
            assert len(winning_candidate_ids) > 0 and len(loss_candidate_ids) > 0
            model_win = np.random.choice(winning_candidate_ids)
            model_loss = np.random.choice(loss_candidate_ids)
            
            model_win_sparse_x0_path = osp.join(prompt_root, f"{image_id:03d}_sparse_sample_{model_win:03d}.pt")
            model_loss_sparse_x0_path = osp.join(prompt_root, f"{image_id:03d}_sparse_sample_{model_loss:03d}.pt")

            model_win_sparse_x0 = torch.load(model_win_sparse_x0_path, weights_only=True, map_location="cpu")
            model_loss_sparse_x0 = torch.load(model_loss_sparse_x0_path, weights_only=True, map_location="cpu")

            return {
                "prompt": prompt,
                "cond": cond,
                "model_win_sparse_x0": model_win_sparse_x0,
                "model_loss_sparse_x0": model_loss_sparse_x0,
            }

    def __len__(self):
        return 100000  # dummy value
    