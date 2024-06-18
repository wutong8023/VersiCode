"""
pass@k Indicator to evaluate the ability of token level
"""

import json
import os
import math

model_name = 'Llama-3-70b-chat-hf'

result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/random_token_data_2024_5_14v2.json'
# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/docstring.json'
# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/respository.json'
# result_path = f'../../dataset/final_dataset/final_generate_token_result/{model_name}/stackoverflow.json'
def compute_score_k(answer:str, model_output:list, k:int):

    c = 0
    n = len(model_output)
    for output in model_output:
        if answer == output:
            c += 1
    if n-c<k:
        return 1.0

    score = 1 - (math.comb(n - c, k))/(math.comb(n, k))

    return score



with open(result_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data = lodict

data_list = data['data']
score_list = []

for d in data_list:
    answer = d['answer']
    model_output_list = eval(d['model_output_token_clear'])#change block or token or line
    temp_score = compute_score_k(answer, model_output_list, 3)
    score_list.append(temp_score)



final_score = sum(score_list)/len(score_list)

print(final_score)
