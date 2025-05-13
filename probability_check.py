import os
import argparse
import random
import pandas as pd
import numpy as np
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
    prompt_user += 'Question: ' + qeury + '\nThought: ' + 'Use information from document: Nanjing was later the capital city of Danyang Prefecture, and had been the capital city of Yangzhou for ...'
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
    final_prompt = deepcopy(prompt_template)
    final_prompt[-1]["content"] = final_prompt[-1]["content"].replace("${prompt_user}", user_prompt)
    return final_prompt

def get_response(retrieved_passages: List[str], query: str):
    llm_config = llm_config_dict()
    params = deepcopy(llm_config['generate_params'])
    params['message'] = get_prompt_with_template(retrieved_passages, query)
    
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

def plot_token_probs(query: str, logprobs: List[tuple], query_num: str, file_name: str):
    tokens = [logprob[0] for logprob in logprobs]
    probs = [logprob[1] for logprob in logprobs]
    
    plt.figure(figsize=(max(6, len(tokens) * 0.5), 4))
    plt.bar(range(len(tokens)), probs)
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
    plt.ylim([0.0, 3.0])
    plt.ylabel("-log p  (Nat)")
    plt.title(query)
    plt.tight_layout()
    
    if query_num:
        query_num = f"./result_plot/{file_name}/query_{query_num}.png"
        query_num = Path(query_num).expanduser()
        query_num.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(query_num, dpi=300, bbox_inches="tight")
        # print(f"[INFO] plot saved! →  {save_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type= int, default= None, dest="query_num", action= "store")
    parser.add_argument("-f", "--file", type= str, default= None, dest="file_name", action= "store")
    parser.add_argument("-a", "--all", dest="all_logprobs", action= "store_true")
    
    args = parser.parse_args()
    file_name = args.file_name
    error_case = pd.read_csv(f"{file_name}.csv")
    error_case_analysis = error_case[['question', 'doc_1', 'doc_2', 'doc_3', 'doc_4', 'doc_5']].copy()
    
    if args.query_num:
        retrieved_passages = [error_case_analysis['doc_1'][args.query_num], 
                                error_case_analysis['doc_2'][args.query_num], 
                                error_case_analysis['doc_3'][args.query_num], 
                                error_case_analysis['doc_4'][args.query_num], 
                                error_case_analysis['doc_5'][args.query_num]
                                ]             
        query = error_case_analysis['question'][args.query_num]
        
        llm_output, logprobs = get_response(retrieved_passages, query)
        print(llm_output, "\n", logprobs)
        plot_token_probs(query, logprobs, str(args.query_num)+"adj2", file_name)
    else:
        if args.all_logprobs:
            all_llm_outputs = []
            all_logprobs = []
            all_mean_logporbs = []
            for i in tqdm(range(len(error_case_analysis)), desc = "Get Logprobs"):
                retrieved_passages = [error_case_analysis['doc_1'][i], 
                                    error_case_analysis['doc_2'][i], 
                                    error_case_analysis['doc_3'][i], 
                                    error_case_analysis['doc_4'][i], 
                                    error_case_analysis['doc_5'][i]
                                    ] 
                query = error_case_analysis['question'][i]
                
                llm_output, logprobs = get_response(retrieved_passages, query)
                
                all_llm_outputs.append(llm_output)
                all_logprobs.append(logprobs)
                probs = [logprob[1] for logprob in logprobs]
                mean_logporbs = np.mean(probs)
                all_mean_logporbs.append(mean_logporbs)
                
            error_case_analysis['llm_outputs'] = all_llm_outputs
            error_case_analysis['logprobs'] = all_logprobs
            error_case_analysis['mean_logporbs'] = all_mean_logporbs
            
            error_case_analysis.to_csv(f"./{file_name}_logprbs.csv")
            print(np.mean(all_mean_logporbs))
                        
        else: 
            sample = sorted(random.sample(range(len(error_case_analysis)), 10))
        
            for i in tqdm(sample, desc= "Get Logprobs"):
                retrieved_passages = [error_case_analysis['doc_1'][i], 
                                    error_case_analysis['doc_2'][i], 
                                    error_case_analysis['doc_3'][i], 
                                    error_case_analysis['doc_4'][i], 
                                    error_case_analysis['doc_5'][i]
                                    ] 
                query = error_case_analysis['question'][i]
                
                llm_output, logprobs = get_response(retrieved_passages, query)
                print(llm_output, "\n", logprobs)
                plot_token_probs(query, logprobs, str(i), file_name)