import json
from llama_cpp import Llama 
from langchain import PromptTemplate 

llm = Llama(model_path="llama3env/Llama3Pipline/models/phi3.gguf",
        n_ctx=1000,  
        n_threads=8,
        n_gpu_layers=35,
        verbose=False)


system_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
create a poster with these specifications: {user_input}. Make sure to include title, background color, background image,
text that should be included in the poster.

"""
)

def llama3(user_input: str)->dict:
    model_input = system_prompt.format(user_input = user_input) 
    response = llm(model_input, max_tokens=512)
    response_text = response['choices'][0]['text']
    return response_text



user_input = "a summer event with beach theme, including sandy background, ocean waves, and event details."
answer = llama3(user_input)

print(answer)