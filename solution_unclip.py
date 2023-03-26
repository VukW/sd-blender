import logging

import click
import numpy as np
import torch
import PIL
from PIL import Image
from diffusers import StableUnCLIPImg2ImgPipeline

logging.root.setLevel(logging.INFO)


class StableUnCLIPImg2ImgPipelinePatched(StableUnCLIPImg2ImgPipeline):
    # this function has a small bug in the original pipeline, thus has to be fixed outside
    def _encode_image(
            self,
            image,
            device,
            batch_size,
            num_images_per_prompt,
            do_classifier_free_guidance,
            noise_level,
            generator,
            image_embeds,
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        if isinstance(image, PIL.Image.Image):
            # the image embedding should repeated so it matches the total batch size of the prompt
            repeat_by = batch_size
        else:
            # assume the image input is already properly batched and just needs to be repeated so
            # it matches the num_images_per_prompt.
            #
            # NOTE(will) this is probably missing a few number of side cases. I.e. batched/non-batched
            # `image_embeds`. If those happen to be common use cases, let's think harder about
            # what the expected dimensions of inputs should be and how we handle the encoding.
            repeat_by = num_images_per_prompt

        if image_embeds is None:
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeds = self.image_encoder(image).image_embeds

        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        image_embeds = image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, repeat_by, 1)
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        image_embeds = image_embeds.squeeze(1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        return image_embeds


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

    pipe = StableUnCLIPImg2ImgPipelinePatched.from_pretrained(repo_id, torch_dtype=torch_dtype, variation="fp16")
    pipe = pipe.to(device)

    embeds1 = pipe.feature_extractor(images=image1, return_tensors="pt").pixel_values
    embeds1 = pipe.image_encoder(embeds1).image_embeds
    embeds2 = pipe.feature_extractor(images=image2, return_tensors="pt").pixel_values
    embeds2 = pipe.image_encoder(embeds2).image_embeds
    imgs = pipe(prompt="",
                image_embeds=(embeds1 + embeds2) / 2,
                num_inference_steps=steps
                ).images
    out_img = imgs[0]
    out_img.save(out_path)
    logging.info(f"{out_path} saved")


if __name__ == '__main__':
    main()
