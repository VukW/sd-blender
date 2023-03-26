import importlib
import logging
from pathlib import Path

import PIL
import click
import clip
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf

logging.root.setLevel(logging.INFO)


def load_model_from_config(config, ckpt, device, verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    #     with all_logging_disabled():
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
    model.to(device)
    if 'cuda' in device:
        model.half()
    model.eval()
    model.cond_stage_model.device = device
    return model


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def to_im_list(x_samples_ddim):
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        ims.append(PIL.Image.fromarray(x_sample.astype(np.uint8)))
    return ims


@torch.no_grad()
def sample(sampler, model, c, uc, scale, start_code, h=320, w=320, ddim_steps=50):
    ddim_eta = 0.0
    precision_scope = torch.autocast
    with precision_scope("cuda"):
        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                         conditioning=c,
                                         batch_size=c.shape[0],
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         eta=ddim_eta,
                                         x_T=start_code)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return to_im_list(x_samples_ddim)


@torch.no_grad()
def get_im_c(im_path, clip_model, preprocess, device):
    # im = Image.open(im_path).convert("RGB")
    prompts = preprocess(im_path).to(device).unsqueeze(0)
    return clip_model.encode_image(prompts).float()


@click.command()
@click.argument('image1_path')
@click.argument('image2_path')
@click.argument('out_path')
@click.option('--device', help="cpu / cuda. If not declared, defined automatically")
@click.option('--scale', default=7.0, type=float, show_default=True, help="guidance scale")
@click.option('--seed', type=int, help="random seed")
@click.option('--steps', default=30, show_default=True, type=int, help="number of steps")
@click.option('--w1', default=1.0, show_default=True, type=float, help="weight of the first image")
@click.option('--w2', default=1.0, show_default=True, type=float, help="weight of the second image")
def main(image1_path: str,
         image2_path: str,
         out_path: str,
         device: str,
         scale: float,
         seed: int,
         steps: int,
         w1: float,
         w2: float):
    if not device:
        device = "cuda" if torch.cuda.is_available else "cpu"
    logging.info(f"`{device}` device would be used.")

    if seed is None:
        seed = np.random.randint(1000000)
    logging.info(f"random seed {seed} would be used")

    ckpt = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-pruned.ckpt")
    config = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")

    model = load_model_from_config(config, ckpt, device=device, verbose=False)
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    h = w = 320

    img1 = PIL.Image.open(image1_path)
    img2 = PIL.Image.open(image2_path)

    torch.manual_seed(seed)
    start_code = torch.randn(1, 4, h // 8, w // 8, device=device)

    conds = [w1 * get_im_c(img1, clip_model, preprocess, device),
             w2 * get_im_c(img2, clip_model, preprocess, device)]

    print([_.shape for _ in conds])
    conds = torch.cat(conds, dim=0).unsqueeze(0)
    sampler = DDIMSampler(model)
    # sampler = PLMSSampler(model)

    imgs = sample(sampler, model, conds, 0 * conds, scale, start_code, ddim_steps=steps)

    out_img = imgs[0]
    out_img.save(out_path)
    logging.info(f"{out_path} saved")


if __name__ == '__main__':
    main()
