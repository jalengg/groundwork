#!/usr/bin/env python
# coding=utf-8
"""SD 1.5 ControlNet trainer — 5-channel conditioning via SD15Dataset."""

import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from data_pipeline.dataset_sd15 import SD15Dataset, sd15_worker_init
from data_pipeline.prep_flux_dataset import PROMPTS

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SD 1.5 ControlNet training with 5-ch conditioning.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_model_name_or_path", type=str, required=True)
    parser.add_argument("--city_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="controlnet-sd15")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--max_train_steps", type=int, default=15000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--validation_image", type=str, default=None, nargs="+")
    parser.add_argument("--num_validation_images", type=int, default=2)
    parser.add_argument("--p_inpaint", type=float, default=0.5)
    parser.add_argument("--min_road_fraction", type=float, default=0.05)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError("`--resolution` must be divisible by 8.")

    return args


def encode_prompt(prompt, tokenizer, text_encoder, device, weight_dtype):
    """Encode a single string prompt; returns (1, seq_len, 768) tensor."""
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(inputs.input_ids.to(device))[0]
    return embeds.to(dtype=weight_dtype)


def log_validation(vae, unet, controlnet, tokenizer, text_encoder, args, accelerator, weight_dtype, step):
    logger.info("Running validation...")
    controlnet_unwrapped = accelerator.unwrap_model(controlnet)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        controlnet=controlnet_unwrapped,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    prompt = PROMPTS["us_suburb"]

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    image_logs = []
    for val_img_path in args.validation_image:
        from PIL import Image
        from torchvision import transforms as T

        val_image = Image.open(val_img_path).convert("RGB")
        val_image = T.Compose([
            T.Resize(args.resolution),
            T.CenterCrop(args.resolution),
        ])(val_image)

        images = []
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                out = pipeline(
                    prompt=prompt,
                    image=val_image,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]
            images.append(out)
        image_logs.append({"control": val_image, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            import numpy as np
            for log in image_logs:
                grid = [np.asarray(log["control"])] + [np.asarray(img) for img in log["images"]]
                grid = np.stack(grid)
                tracker.writer.add_images(f"validation/step_{step}", grid, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            wandb_images = []
            for log in image_logs:
                wandb_images.append(wandb.Image(log["control"], caption="control"))
                for img in log["images"]:
                    wandb_images.append(wandb.Image(img, caption=prompt))
            tracker.log({"validation": wandb_images}, step=step)

    del pipeline
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return image_logs


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir) / args.logging_dir
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load models ---
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Cache prompt embedding once
    cached_encoder_hidden_states = encode_prompt(
        PROMPTS["us_suburb"], tokenizer, text_encoder, accelerator.device, weight_dtype
    )

    # --- Register accelerate save/load hooks for controlnet ---
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            i = len(weights) - 1
            while len(weights) > 0:
                weights.pop()
                models[i].save_pretrained(os.path.join(output_dir, "controlnet"))
                i -= 1

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            loaded = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
            model.register_to_config(**loaded.config)
            model.load_state_dict(loaded.state_dict())
            del loaded

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # --- Dataset + DataLoader ---
    dataset = SD15Dataset(
        city_dirs=args.city_dirs,
        augment=True,
        p_inpaint=args.p_inpaint,
        min_road_fraction=args.min_road_fraction,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=sd15_worker_init,
    )

    # --- Optimizer + LR scheduler ---
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
    )

    controlnet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = {k: v for k, v in vars(args).items() if not isinstance(v, list)}
        accelerator.init_trackers("sd15_train_controlnet", config=tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # --- Auto-resume from latest checkpoint ---
    checkpoints = sorted(
        [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    ) if os.path.isdir(args.output_dir) else []

    if checkpoints:
        latest = checkpoints[-1]
        accelerator.print(f"Resuming from checkpoint {latest}")
        accelerator.load_state(os.path.join(args.output_dir, latest))
        global_step = int(latest.split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, num_train_epochs):
        controlnet.train()
        for cond_5ch, target_3ch in dataloader:
            with accelerator.accumulate(controlnet):
                # Encode target to latents
                target_pixels = target_3ch.to(dtype=torch.float32) * 2.0 - 1.0
                with torch.no_grad():
                    latents = vae.encode(target_pixels).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(
                    latents.float(), noise.float(), timesteps
                ).to(dtype=weight_dtype)

                controlnet_cond = cond_5ch.to(dtype=weight_dtype)

                # Expand cached embedding to batch size
                encoder_hidden_states = cached_encoder_hidden_states.expand(bsz, -1, -1)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[s.to(dtype=weight_dtype) for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # Rolling checkpoint eviction
                        existing = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(existing) >= args.checkpoints_total_limit:
                            to_remove = existing[: len(existing) - args.checkpoints_total_limit + 1]
                            for ckpt in to_remove:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt))
                                logger.info(f"Removed old checkpoint: {ckpt}")

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                    if (
                        args.validation_image is not None
                        and global_step % args.validation_steps == 0
                    ):
                        log_validation(
                            vae=vae,
                            unet=unet,
                            controlnet=controlnet,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet_final = accelerator.unwrap_model(controlnet)
        controlnet_final.save_pretrained(args.output_dir)
        logger.info(f"Saved final ControlNet to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
