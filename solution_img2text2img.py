import logging

import click
import numpy as np
import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import more_itertools

logging.root.setLevel(logging.INFO)


@click.command()
@click.argument('image1_path')
@click.argument('image2_path')
@click.argument('out_path')
@click.option('--device', help="cpu / cuda. If not declared, defined automatically")
@click.option('--seed', type=int, help="random seed")
@click.option('--steps', default=30, show_default=True, type=int, help="number of steps")
def main(image1_path: str,
         image2_path: str,
         out_path: str,
         device: str,
         seed: int,
         steps: int):
    if not device:
        device = "cuda" if torch.cuda.is_available else "cpu"
    logging.info(f"`{device}` device would be used.")

    if seed is None:
        seed = np.random.randint(1000000)
    logging.info(f"random seed {seed} would be used")

    torch.manual_seed(seed)
    interrogator_config = Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k")
    interrogator_config.apply_low_vram_defaults()
    interrogator_config.device = 'cpu'
    ci = Interrogator(interrogator_config)

    image1 = Image.open(image1_path).convert('RGB')
    prompt1 = ci.interrogate_fast(image1)
    logging.info(f'first image description: {prompt1}')
    image2 = Image.open(image2_path).convert('RGB')
    prompt2 = ci.interrogate_fast(image2)
    logging.info(f'second image description: {prompt2}')

    combined_prompt = ', '.join(list(more_itertools.interleave_longest(prompt1.split(', '), prompt2.split(', '))))
    logging.info(f"combined prompt: {combined_prompt}")

    repo_id = "stabilityai/stable-diffusion-2-base"
    if 'cuda' in device:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    imgs = pipe(combined_prompt, num_inference_steps=steps).images
    out_img = imgs[0]
    out_img.save(out_path)
    logging.info(f"{out_path} saved")


if __name__ == '__main__':
    main()
