from copy import deepcopy
import datetime
from glob import glob
import inspect
import logging
import math
import os
from typing import Optional, Union, List

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import argparse
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers

from trellis import models
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

from dataset import SyntheticDataset

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__, log_level="INFO")

os.environ["WANDB_API_KEY"] = "ce1816bb45de1fc10f92c8bc17f2d7cc9b1a8757"


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def get_model():
    sparse_structure_flow_model = models.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16",
    )
    return sparse_structure_flow_model


def create_output_folders(output_dir, config, exp_name):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"{exp_name}_{now}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))
    return out_dir


def sample_flow_matching_t_for_training(logit_normal_mu, logit_normal_sigma, bsz):
    """
    Sample t for flow matching training with a LogitNormal distribution.
    """
    t = torch.randn(bsz)
    t = torch.sigmoid(logit_normal_mu + t * logit_normal_sigma)
    return t


def forward_flow_matching_loss(model, x0, t, cond, eps=None, **kwargs):
    if eps is None:
        eps = torch.randn_like(x0)
    xt = (1 - t.view(t.shape + (1,) * (x0.ndim - 1))) * x0 + t.view(t.shape + (1,) * (x0.ndim - 1)) * eps
    target = eps - x0
    pred = model(xt, t * 1000, cond, **kwargs)
    loss = (pred - target).pow(2).mean(dim=tuple(range(1, x0.ndim)))
    return loss


def forward_dpo_loss(model, ref_model, x0_win, x0_loss, t, cond, beta, sample_same_epsilon, **kwargs):
    # 0. Concatenate x0_win and x0_loss
    x0 = torch.cat([x0_win, x0_loss], dim=0)
    t = torch.cat([t, t], dim=0)
    cond = torch.cat([cond, cond], dim=0)

    # 1. Forward pass
    eps = torch.randn_like(x0)
    loss_w, loss_l = forward_flow_matching_loss(model, x0, t, cond, eps, **kwargs).chunk(2)
    with torch.no_grad():
        loss_w_ref, loss_l_ref = forward_flow_matching_loss(ref_model, x0, t, cond, eps=eps if sample_same_epsilon else None, **kwargs).detach().chunk(2)

    model_diff = loss_w - loss_l
    ref_diff = loss_w_ref - loss_l_ref

    inside_term = -0.5 * beta * (model_diff - ref_diff)
    loss = -F.logsigmoid(inside_term)
    return loss.mean()


def forward_dro_loss(model, x0_win, x0_loss, t, cond, **kwargs):
    # 0. Concatenate x0_win and x0_loss
    x0 = torch.cat([x0_win, x0_loss], dim=0)
    t = torch.cat([t, t], dim=0)
    cond = torch.cat([cond, cond], dim=0)

    # 1. Forward pass
    eps = torch.randn_like(x0)
    loss_w, loss_l = forward_flow_matching_loss(model, x0, t, cond, eps, **kwargs).chunk(2)
    
    model_diff = loss_w - loss_l
    loss = model_diff.mean()
    return loss.mean()


def forward_sft_loss(model, x0, t, cond, **kwargs):
    eps = torch.randn_like(x0)
    loss = forward_flow_matching_loss(model, x0, t, cond, eps, **kwargs)
    return loss.mean()


def main_eval(
    image_paths: Union[str, List[str]],
    use_original: bool = False,
    ckpt_path: str = None,
    num_models_per_image: int = 16,
    sample_dir: str = "./samples",
    naming_level: int = 1,
    **kwargs,
):
    assert ckpt_path is not None or use_original, "Either ckpt_path or use_original must be provided"
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    if not use_original:
        run_config = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml")
        run_config = OmegaConf.load(run_config)
        if run_config.use_lora:
            state_dict = load_file(ckpt_path)
            peft_config = LoraConfig(
                r=run_config.lora_r,
                lora_alpha=run_config.lora_alpha,
                lora_dropout=run_config.lora_dropout,
                target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
            )
            pipeline.models["sparse_structure_flow_model"] = get_peft_model(pipeline.models["sparse_structure_flow_model"], peft_config)
            pipeline.models["sparse_structure_flow_model"].load_state_dict(state_dict)
        else:
            state_dict = load_file(ckpt_path)
            sparse_structure_flow_model = pipeline.models["sparse_structure_flow_model"]
            sparse_structure_flow_model.load_state_dict(state_dict)
            
    os.makedirs(sample_dir, exist_ok=True)

    bsz, num_batches = 1, num_models_per_image

    if isinstance(image_paths, str):
        image_paths = sorted(glob(image_paths))
    else:
        image_paths = sorted(image_paths)

    stable_object_id_fpath = "./objaverse-eval/stable_objaverse_ids.txt"
    stable_object_ids = [line.strip() for line in open(stable_object_id_fpath).read().splitlines()]

    for image_path in tqdm(image_paths):
        if not any([oid in image_path for oid in stable_object_ids]):
            continue

        num_existing_glbs = len(glob(os.path.join(sample_dir, f"{'-'.join(image_path.split('/')[-naming_level:]).replace('.jpg', '_*.glb')}")))
        if num_existing_glbs >= num_models_per_image:
            continue
        num_existing_batches = num_existing_glbs // bsz

        image = Image.open(image_path)        
        image = pipeline.preprocess_image(image)
        
        for bid in range(num_existing_batches, num_batches):
            seed = bid

            try:
                outputs, _, _, _ = pipeline.run(
                    image,
                    seed=seed + 1,
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
                )
            except Exception as e:
                print(e)
                continue

            for i in range(bsz):
                eid = bid * bsz + i
                glb_path = os.path.join(sample_dir, f"{'-'.join(image_path.split('/')[-naming_level:]).replace('.jpg', f'_{eid:03d}.glb')}").replace(".png", f"_{eid:03d}.glb")
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][i],
                    outputs['mesh'][i],
                    simplify=0.95,          # Ratio of triangles to remove in the simplification process
                    texture_size=1024,      # Size of the texture used for the GLB
                    with_texture=False,     # Disable texture for faster stability evaluation
                )
                glb.export(glb_path)
        
    print(f"Saved samples to {sample_dir}")


def main(
    exp_name: str,
    output_dir: str = "./runs",
    dataset_dir: str = "./data",
    dataset_kwargs: dict = {},
    category: str = "clock",
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 2000,
    use_adafactor: bool = False,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 0.0,
    adam_epsilon: float = 1e-8,
    max_train_steps: int = 10000,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,

    flow_matching_t_logit_normal_mu: float = 1.0,
    flow_matching_t_logit_normal_sigma: float = 1.0,
    dpo_beta: float = 1.0,
    sample_same_epsilon: bool = True,

    log_interval: int = 10,
    save_interval: int = 100,
    ckpt_interval: int = 5000,
    seed: Optional[int] = None,
    logger_type: str = "tensorboard",
    resume_from_checkpoint: Optional[str] = None,

    use_lora: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.0,

    use_sft: bool = False,
    use_dro: bool = False,
    use_dpo: bool = False,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    assert use_dpo + use_sft + use_dro == 1, "Only one of use_sft, use_dro, or use_dpo can be True"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=logger_type,
        project_dir=output_dir,
    )
    create_logging(logging, logger, accelerator)
    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        run_dir = create_output_folders(output_dir, config, exp_name)

    if scale_lr:
        learning_rate = learning_rate * accelerator.num_processes * gradient_accumulation_steps * batch_size

    model = get_model()
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
        )
        model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, 
        betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon
    ) if not use_adafactor else transformers.Adafactor(
        model.parameters(), lr=learning_rate, eps=adam_epsilon, weight_decay=adam_weight_decay,
        clip_threshold=1.0, scale_parameter=False, relative_step=False
    )

    lr_scheduler = get_scheduler(
        "constant_with_warmup", optimizer=optimizer, 
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    train_dataset = SyntheticDataset(
        root_dir=dataset_dir,
        category=category,
        **dataset_kwargs,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
    if use_dpo:
        ref_model = deepcopy(model)
        ref_model.requires_grad_(False)

    if accelerator.is_main_process:
        accelerator.init_trackers(project_name="dso", config=config, init_kwargs={"wandb": {"name": exp_name}})

    total_batch_size = accelerator.num_processes * gradient_accumulation_steps * batch_size
    num_train_epochs = math.ceil(max_train_steps * gradient_accumulation_steps/ len(train_loader))

    if resume_from_checkpoint is not None:
        global_step = int(resume_from_checkpoint.split("-")[-1])
        accelerator.load_state(resume_from_checkpoint, strict=False)
    else:
        global_step = 0

    logger.info(f"Model loaded! {model}, num params: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(0, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Steps")
    progress_bar.update(global_step)

    running_loss = 0.0
    while global_step < max_train_steps:
        for batch in train_loader:
            bsz = batch['cond'].shape[0]
            t = sample_flow_matching_t_for_training(
                logit_normal_mu=flow_matching_t_logit_normal_mu,
                logit_normal_sigma=flow_matching_t_logit_normal_sigma,
                bsz=bsz,
            ).to(batch['cond'])

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    if use_dpo:
                        loss = forward_dpo_loss(
                            model=model,
                            ref_model=ref_model,
                            x0_win=batch['model_win_sparse_x0'],
                            x0_loss=batch['model_loss_sparse_x0'],
                            t=t,
                            cond=batch['cond'],
                            beta=dpo_beta,
                            sample_same_epsilon=sample_same_epsilon,
                        )
                    elif use_dro:
                        loss = forward_dro_loss(
                            model=model,
                            x0_win=batch['model_win_sparse_x0'],
                            x0_loss=batch['model_loss_sparse_x0'],
                            t=t,
                            cond=batch['cond'],
                        )
                    else:  # use_sft
                        loss = forward_sft_loss(
                            model=model,
                            x0=batch['model_win_sparse_x0'],
                            t=t,
                            cond=batch['cond'],
                        )

                avg_loss = accelerator.gather(loss).mean()
                running_loss += avg_loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % log_interval == 0 and accelerator.is_main_process:
                    log_dict = dict(loss=running_loss / log_interval / gradient_accumulation_steps, step=global_step)
                    accelerator.log(log_dict, step=global_step)
                    running_loss = 0.0
                
                if global_step % save_interval == 0 and accelerator.is_main_process:
                    pass

                if global_step % ckpt_interval == 0 and accelerator.is_main_process:
                    save_path = os.path.join(run_dir, f"checkpoint-{global_step:07d}")
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(save_path)
            
                if global_step >= max_train_steps:
                    break
            
            logs = dict(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
            progress_bar.set_postfix(**logs)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    
    # Load config
    args_dict = OmegaConf.load(args.config)
    if args.eval:
        main_eval(**args_dict)
    else:
        main(**args_dict)
