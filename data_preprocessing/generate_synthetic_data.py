from glob import glob
import os

import argparse
from PIL import Image
import torch
from tqdm import tqdm
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.


from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=0)
parser.add_argument("--num_jobs", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--with_texture", action="store_true")
parser.add_argument("--save_extra", action="store_true")
parser.add_argument("--output_dir", type=str, default="/mnt/disks/data/DSO-synthetic-data")
parser.add_argument("--image_paths", type=str, required=True)

args = parser.parse_args()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

sample_batch_size, num_batches = 1, args.num_samples

if isinstance(args.image_paths, str):
    images = sorted(glob(args.image_paths))
else:
    images = sorted(args.image_paths)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
print(f"################ Total images to generate: {len(images)} ################")

for image_path in tqdm(images[args.job_id::args.num_jobs]):
    image = Image.open(image_path)
    image = pipeline.preprocess_image(image)

    category, prompt, image_name = image_path.split("/")[-3:]

    save_root = os.path.join(output_dir, f"{category}/{prompt}")
    os.makedirs(save_root, exist_ok=True)
    generate_3d = True
    if len(glob(os.path.join(save_root, f"{image_name.replace('.jpg', '_*.glb').replace('.png', '_*.glb')}"))) >= args.num_samples:
        generate_3d = False

    if generate_3d:
        for bid in range(num_batches):
            # Check if all samples in this batch already exist
            skip_batch = True
            for i in range(sample_batch_size):
                eid = bid * sample_batch_size + i
                glb_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_{eid:03d}.glb').replace('.png', f'_{eid:03d}.glb')}")
                sparse_sample_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_sparse_sample_{eid:03d}.pt').replace('.png', f'_sparse_sample_{eid:03d}.pt')}")
                slat_sample_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_slat_sample_{eid:03d}.pt').replace('.png', f'_slat_sample_{eid:03d}.pt')}")
                if not os.path.exists(glb_path) or not os.path.exists(sparse_sample_path) or not os.path.exists(slat_sample_path):
                    skip_batch = False
                    break
            
            if skip_batch:
                continue

            success = False
            # Run the pipeline
            try:
                outputs, cond, sparse_x0, slat = pipeline.run(
                    image,
                    seed=bid,
                    num_samples=sample_batch_size,
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": 12,
                        "cfg_strength": 7.5,
                    },
                    slat_sampler_params={
                        "steps": 12,
                        "cfg_strength": 3,
                    },
                )
                success = True

            except Exception as e:
                print(e)
                pass
            
            if not success:
                print(f"Failed to generate 3D models for {image_path}")
                continue

            if args.save_extra:
                if bid == 0:
                    # Cache the image cond
                    image_cond = cond["cond"][0].to(dtype=torch.bfloat16)
                    image_cond_save_path = os.path.join(save_root, f"{image_name.replace('.jpg', '_cond.pt').replace('.png', '_cond.pt')}")
                    torch.save(image_cond.cpu(), image_cond_save_path)

                for i in range(sample_batch_size):
                    eid = bid * sample_batch_size + i
                    sparse_sample = sparse_x0[i].to(dtype=torch.bfloat16)
                    sparse_sample_save_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_sparse_sample_{eid:03d}.pt').replace('.png', f'_sparse_sample_{eid:03d}.pt')}")
                    torch.save(sparse_sample.cpu(), sparse_sample_save_path)

            # GLB files can be extracted from the outputs
            for i in range(sample_batch_size):
                glb_path = os.path.join(save_root, f"{image_name.replace('.jpg', f'_{(bid*sample_batch_size+i):03d}.glb').replace('.png', f'_{(bid*sample_batch_size+i):03d}.glb')}")
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][i],
                    outputs['mesh'][i],
                    simplify=0.95,          # Ratio of triangles to remove in the simplification process
                    texture_size=1024,      # Size of the texture used for the GLB
                    with_texture=args.with_texture,
                )
                glb.export(glb_path)
