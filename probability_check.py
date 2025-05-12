import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
from tqdm import tqdm
from prompt_template import prompt_template


def template_creation(retrieved_passages: List[str], qeury):
    prompt_user = ''
    for passage in retrieved_passages:
        prompt_user += f'Wikipedia Title: {passage}\n\n'
    prompt_user += 'Question: ' + qeury + '\nThought: ' # + 'Use information from document: Nanjing was later the capital city of Danyang Prefecture, and had been the capital city of Yangzhou for about 400 years from late Han to early Tang'
    return prompt_user

def llm_config_dict():
    config_dict = {}
    config_dict['llm_name'] = "gpt-4o-mini"
    config_dict['llm_base_url'] = "https://api.openai.com/v1"
    config_dict['generate_params'] = {
        "model": "gpt-4o-mini",
        "max_completion_tokens": 2048,
        "n": 1,
        "seed": 0,
        "temperature": 0.0,
        "logprobs": True,
        "top_log_probs": 1
    }
    return config_dict

def get_prompt_with_template(retrieved_passages: List[str], query: str):
    user_prompt = template_creation(retrieved_passages, query)
    final_prompt = prompt_template.copy()
    final_prompt[-1]["content"] = final_prompt[-1]["content"].replace("${prompt_user}", user_prompt)
    return final_prompt

def get_response(retrieved_passages: List[str], query: str):
    llm_config = llm_config_dict()
    llm_config['generate_params']['message'] = get_prompt_with_template(retrieved_passages, query)
    params = deepcopy(llm_config['generate_params'])
    
    load_dotenv()
    client = OpenAI(base_url= llm_config['llm_base_url'], api_key= os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(model= params['model'],
                                              messages= params['message'],
                                              max_completion_tokens= params['max_completion_tokens'],
                                              n= params['n'],
                                              seed= params['seed'],
                                              temperature= params['temperature'],
                                              logprobs= params['logprobs'],
                                              top_logprobs= params['top_log_probs'],
                                              )

    llm_output = response.choices[0].message.content
    logprobs = []
    for token_data in response.choices[0].logprobs.content:
        logprobs.append((token_data.token, -1 * token_data.logprob))
        
    return llm_output, logprobs

def plot_token_probs(query: str, logprobs: List[tuple], save_path: str):
    tokens = [logprob[0] for logprob in logprobs]
    probs = [logprob[1] for logprob in logprobs]
    
    plt.figure(figsize=(max(6, len(tokens) * 0.5), 4))
    plt.bar(range(len(tokens)), probs)
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
    plt.ylim([0.0, 3.0])
    plt.ylabel("-log p  (Nat)")
    plt.title(query)
    plt.tight_layout()
    
    if save_path:
        save_path = f"./result_plot/query_{save_path}.png"
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # print(f"[INFO] plot saved! →  {save_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type= int, default= 0, dest="query_num", action= "store")
    
    args = parser.parse_args()
    error_case = pd.read_csv("hard_error_case.csv")
    error_case_analysis = error_case[['question', 'doc_1', 'doc_2', 'doc_3', 'doc_4', 'doc_5']]
    
    for i in tqdm(range(len(error_case_analysis)), desc= "Get Logprobs"):
        retrieved_passages = [error_case_analysis['doc_1'][i], 
                            error_case_analysis['doc_2'][i], 
                            error_case_analysis['doc_3'][i], 
                            error_case_analysis['doc_4'][i], 
                            error_case_analysis['doc_5'][i]
                            ] 
        
        query = error_case_analysis['question'][i]
        llm_output, logprobs = get_response(retrieved_passages, query)
        plot_token_probs(query, logprobs, str(i))