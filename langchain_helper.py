import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import pandas as pd
from smolagents import CodeAgent, TransformersModel
import matplotlib.pyplot as plt
import seaborn as sns

def get_agent():
    ##--- load data ---##
    df = pd.read_csv('/home/lero/idrive/cmac/DDMAP/Stability studies/llm_dataset.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    ## --- load model ---##
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

    model = TransformersModel(
        model_id=model_id,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        max_new_tokens=512,
        temperature=0.2
    )


    custom_instructions = """
    You are a pharmaceutical data analyst.
    You are working with a dataframe called 'df'.
    The columns are: 'stable_or_not', 'api', 'polymer', 'drug_loading_(wt%)', 'temperature_(degrees_celcius)', 'humidity_(%_relative_humidity)'.

    IMPORTANT: When filtering for a specific API or Polymer:
    1. Always specify the column, e.g., df[df['API'] == 'name']
    2. Case sensitivity matters: all API or Polymer have a capitalised first letter.
    3. Only make a plot/ visuallisation if the the user asks for one and use 'plt' and 'sns' which are already provided.

    If a question is not related to the provided dataset. Simply say that you will not answer these questions.
    """
    ##--- create agent ---##
    agent = CodeAgent(
        tools=[], 
        model=model, 
        additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn"],
        max_steps=1 
    )

    agent.prompt_templates["system_prompt"] = custom_instructions
    
    return agent, df

# if __name__ == "__main__":
#     agent = get_agent()




