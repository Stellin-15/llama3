from llama3 import llama3_model
from stablediffusion import generate_image


def create_poster(user_input:str)->str:
    poster_components = llama3_model(user_input)
    background_image = poster_components["poster_backgroundImage"]["background image"]
    image_path = generate_image(background_image)
    poster_text = poster_components["poster_text"]
    poster_title = poster_components["poster_title"]

    return image_path,poster_text,poster_title

# here we write main function and return back the title, image path and text for the poster.


