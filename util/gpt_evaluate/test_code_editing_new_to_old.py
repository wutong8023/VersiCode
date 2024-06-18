"""
GPT test code editing task, new version to old version
"""
import json
import openai
from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()
model_name = "gpt-4o"


def predict(content, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        frequency_penalty=0.1,
        max_tokens=512,
        logit_bias=None,
        logprobs=None,
        n=6,
        presence_penalty=0.0,
        seed=None,
        stop=None,
        stream=False,
        temperature=0.8,
        top_p=0.95
    )
    ans_list = []
    choices_list = response.choices
    for c in choices_list:
        content = c.message.content
        ans_list.append(content)
    final_ans = str(ans_list)

    return final_ans


def bulid_prompt(description, old_version, new_code, new_version) -> str:
    """
    construct prompt
    :param version:
    :param description:
    :param masked_code:
    :param options:
    :return:
    """
    prompt = f"""
    You are now a professional Python programming engineer. I will provide you with a code snippet and a description of its functionality, 
    including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified old version. 
    Your task is to refactor the code using the methods provided by the specified old version and return the refactored code. 
    Please note that you only need to return the refactored code and enclose it with <start> and <end>:
    ###Functionality description of the code
    {description}
    ###Dependency and origin version
    {new_version}
    ###Old version code
    {new_code}
    ###Dependency and old version
    {old_version}
    ###Refactored new code
    """

    return prompt


json_path = '../../dataset/final_dataset/code_editing/test_without_docstring.json'


with open(json_path, 'r', encoding='utf-8') as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']


for data in data_list:
    print(f"正在预测第{data_list.index(data) + 1}条")
    old_version = data['dependency'] + data['old_version']  # package == x.x.x
    new_version = data['dependency'] + data['new_version']  # package == x.x.x
    description = data['description']  # function description
    new_code = data['new_code']

    instruction = bulid_prompt(description, old_version, new_code, new_version)
    prediction = predict(instruction, model_name)
    data['model_output'] = prediction

save_folder_path = os.path.join('../../dataset/final_dataset/code_editing_result_nov2', model_name)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8') as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)

