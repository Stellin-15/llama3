import json
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# import all your models from the following file located outside this folder.

import sys
sys.path.insert(0, '/home/v_ubuntu/models')
from model_setup import ModelManager #type: ignore




# The function generate_image takes the following parameters:
# prompt, extra_prompt, negative_prompt, height=768, width=768, num_inference_steps=20, guidance_scale=7.5
# and it returns the generated image path but this function won't work for now 
# because of the gpu thing:(



# initialize your llm using the following line
manager = ModelManager()
model = manager.setup_phi()

class Poster(BaseModel):
    title: str = Field(description="The title of the poster")
    text: list[str] = Field(description="The main text or message of the poster")
    background_image: str = Field(description="URL or path to the background image for the poster")
    background_color: list[str] = Field(description="The background color of the poster, specified in any valid CSS color format")
    fonts: list[str] = Field(description="A list of fonts to be used in the poster")



parser = JsonOutputParser(pydantic_object=Poster)



system_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
{user_input}. 
Make sure to include title, background color(give me atleast 3), background image,
text(give me atleast 5)),fonts(give me atleat 3) aswell that should be included in the poster.\n{format_instructions}\n just give the output in this format no other additional text pls
""",
partial_variables={"format_instructions": parser.get_format_instructions()}
)


#use the following pattern to make the code concise 

def llama3(user_input: str)->str:

    model_input = system_prompt.format(user_input=user_input) 
    response = manager.phi3(model_input)
    parsed_reponse = parser.parse(response)
    return str(parsed_reponse)
    # response = llm(model_input, max_tokens=512)
    # response_text = response['choices'][0]['text']
    # return response_text




user_input = "create a poster for a summer event with beach theme, including sandy background, ocean waves, and event details."
answer = llama3(user_input)
print(answer)