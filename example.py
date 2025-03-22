from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

ckpt_path = "./DSO-finetuned-TRELLIS/dro-4000iters.safetensors"  # "./DSO-finetuned-TRELLIS/dpo-8000iters.safetensors"
image_path = "./image-to-3D-eval-stability-under-gravity/clock-eval/01.jpg"  # Or your own image

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
)
pipeline.models["sparse_structure_flow_model"] = get_peft_model(pipeline.models["sparse_structure_flow_model"], peft_config)
pipeline.models["sparse_structure_flow_model"].load_state_dict(load_file(ckpt_path))

image = Image.open(image_path)
image = pipeline.preprocess_image(image)

outputs = pipeline.run(
    image,
    seed=0,
    preprocess_image=False,
    formats=["gaussian", "mesh"],
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)[0]

glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
    with_texture=True,      # Disable texture for faster stability evaluation
)
glb.export("./output.glb")
