import json
import torch
from langchain import LLMChain, PromptTemplate
from transformers import LlamaForCasualLM,LLamaTokenizer

model_path = ""

model = LlamaForCasualLM.from_pretrained(model_path)
tokenizer = LLamaTokenizer.from_pretained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)

system_prompt = PromptTemplate(input_variables={"user_input"},
    template = """

create a poster with these specifications : {user_input} and make sure to include title,background color,background image,
text that should be included in the poster. All of this should be returned in the following JSON format:

"poster_title" : {
    "title" : 
}
"poster_text" : {
    "text":
}
"poster_backgroundColor":{
    "backgound color": 
    }
"poster_backgroundImage":{
    "background image": 
}


"""                 
 )

def llama3_model(user_input: str)->dict:
    
    model_input = system_prompt.format(user_input = user_input) 

    model_input = tokenizer(model_input,return_tensors = 'pt').to(device)
    model_output = model.generate(model_input)
    response = tokenizer.decode(model_output[0],skip_special_token=True)

    lines = response.split("\n")
    if len(lines)>0:
        title = lines[0].strip()
    if len(lines)>1:
        text = lines[1].strip()
    if len(lines)>2:
        background_color = lines[2].strip()
    if len(lines)>3:
        background_image = lines[3].strip()


    output = {
        "poster_title": {"title": title},
        "poster_text": {"text": text},
        "poster_backgroundColor": {"background color": background_color},
        "poster_backgroundImage": {"background image": background_image}
    }

    return output