# 1, Stable Diffusion by Lambdalabs. Solution based on images conditioning (2 / 5)
The code and model are taken from https://huggingface.co/spaces/lambdalabs/image-mixer-demo ([git](https://github.com/justinpinkney/stable-diffusion#image-mixer))

```shell
python solution_img_conditioning.py --device cpu --seed 1100 --steps 60 \
gigachad.jpg shrek.png gigachad_shrek.png
```

The model is a finetuned version of sd-1.5, that's able to receive CLIP image embeddings as conditionings (instead of CLIP text embeddings)

Shrek + gigachad:

<img src="shrek.png" alt= "" height="320"> <img src="gigachad.jpg" alt= "" height="320">

=>

<img src="gigachad_shrek.png" alt= "" height="320">


Cat + bread:


<img src="cat_yawning.jpg" alt= "" height="320"> <img src="bread.png" alt= "" height="320">

=>

<img src="cat_bread.png" alt= "" height="320">


# 2, Image2text2image. Solution based on images conditioning (1 / 5)

- https://github.com/pharmapsychotic/clip-interrogator is used for generating text prompt by image;
- prompts are combined bax with mixing (`(A, B, C) + (X, Y, Z) => (A, X, B, Y, C, Z)`)
- new image is generated with [SD 2 official base model](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion_2)

```shell
python solution_img2text2img.py --device cpu --seed 1200 --steps 60 \
gigachad.jpg shrek.png gigachad_shrek_img2text2img.png
```

Shrek + gigachad:

<img src="shrek.png" alt= "" height="320"> <img src="gigachad.jpg" alt= "" height="320">

=>

<img src="gigachad_shrek_img2text2img.png" alt= "" height="320">


Cat + bread:


<img src="cat_yawning.jpg" alt= "" height="320"> <img src="bread.png" alt= "" height="320">

=>

<img src="cat_bread_img2text2img.png" alt= "" height="320">


