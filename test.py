import json
import torch
from langchain import LLMChain, PromptTemplate
from transformers import LlamaForCasualLM,LLamaTokenizer

model_path = "" # here you will add the path, once the model is installed in the server, should be in the same directory.

model = LlamaForCasualLM.from_pretrained(model_path)
tokenizer = LLamaTokenizer.from_pretained(model_path)


# To check for GPU- veehive has tesla gpu very good should use it, once they make virtual machine for up 
#  - cuda means that you are searching for a GPU to run the request for the model on, if it available run it
# - else its chill run on cpu 
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)

# now to define the system prompt, the following things have to ne included: 
# -specify the output componets that should be hard coded and that should be included with the user prompt.
# -the output format should be JSON so we can easily pass the components for the next model, in our case that is 
# stable diffusion to generate random images for the poster and send it to front end 

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


# make function to call our model and get a response

def llama3(user_input: str)->dict:
    # final form of the prompt after adding both the system and user prompt
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

    # to test the output generated on the server 

    output = {
        "poster_title": {"title": title},
        "poster_text": {"text": text},
        "poster_backgroundColor": {"background color": background_color},
        "poster_backgroundImage": {"background image": background_image}
    }


    return output



# test run example 
user_input = "Design a poster for a summer event with beach theme, including sandy background, ocean waves, and event details."
answer = llama3(user_input)

print(answer)

    

