import torch 
from diffusers import StableDiffusionPipeline


stable_diffusion_model_path = "" # where you enter the path once the stable diffusion model in the server 

pipe = StableDiffusionPipeline.from_pretrained(stable_diffusion_model_path)

pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def generate_image(background_image: str)->str:

    image = pipe(background_image).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path