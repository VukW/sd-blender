import logging

import click
import numpy as np
import torch
from PIL import Image
from diffusers import StableUnCLIPImg2ImgPipeline
logging.root.setLevel(logging.INFO)


def encode_image(pipe, image):
    with torch.no_grad():
        image = np.asarray([np.asarray(image)])
        image = image / 255.0
        image = torch.Tensor(image).permute(0, 3, 1, 2)
        image = (image - 0.5) * 2
        latents = pipe.vae.encode(image).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        return latents


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

    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    repo_id = "stabilityai/stable-diffusion-2-1-unclip"
    if 'cuda' in device:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, variation="fp16")
    pipe = pipe.to(device)

    imgs = pipe(prompt="", latents=encode_image(pipe, image1),
                image=image2, num_inference_steps=steps).images[0].show()
    out_img = imgs[0]
    out_img.save(out_path)
    logging.info(f"{out_path} saved")


if __name__ == '__main__':
    main()
