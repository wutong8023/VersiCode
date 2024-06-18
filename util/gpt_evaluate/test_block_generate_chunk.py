"""
GPT performs token level generation prediction and truncates overly long tokens
"""
import json
import openai
from openai import OpenAI
import os
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_tokens = 15000   #gpt3.5 is 16ktoken    gpt4o is 128k
model_name = "gpt-3.5-turbo"

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()

def truncate_text_with_tokenizer(text, max_tokens):
    tokens = tokenizer.encode(text)
    print(len(tokens))
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

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


def bulid_prompt(version, description) -> str:
    """
    construct prompt
    :param version:
    :param description:
    :param masked_code:
    :param options:
    :return:
    """
    prompt = f'''
            You are a professional Python engineer, and I will provide functional descriptions and versions of specified dependency packages. 
            You need to write code in Python to implement this feature based on the functional description and using the dependency package and version I specified. 
            Please note that you only need to return the code that implements the function, and do not return any other content. 
            Please use <start> and <end> to enclose the generated code. Here is an example:
            ###Function Description：
            The function of this code is to print the results predicted by calling the model using vllm.
            ###dependeny and version：
            vllm==0.3.3
            ###response:
            <start>
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print("Prompt,Generated text")
            <end>

            ###Function Description：
            {description}
            ###dependeny and version：
            {version}
            ###response:


        '''
    return prompt



json_path = '../../dataset/final_dataset/all_block_data/random_block_data_2024_5_14v2.json' #path of test data
# json_path = '../../dataset/final_dataset/generate_block_result_data2024_5_14v2/gpt-3.5-turbo/random_block_data_2024_5_14v2.json'  #After an error occurs, use the address to continue predicting from where the error occurred


with open(json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']


#   Predicting item by item
for data in data_list:
    if "model_output" in data:
        print(f"the {data_list.index(data) + 1} has already been predicted, skipping this data!")
        continue
    try:
        print(f"Predicting {data_list.index(data) + 1} ")
        version = data['dependency'] + data['version']  #   package == x.x.x
        description = data['description']   #   function description

        instruction = bulid_prompt(version, description)

        truncated_text = truncate_text_with_tokenizer(instruction, max_tokens)
        prediction = predict(truncated_text, model_name)

        data['model_output'] = prediction
    except Exception as e:
        print(f"error：{e}")
        print("save current data")
        save_folder_path = os.path.join('../../dataset/final_dataset/generate_block_result_data2024_5_14v2', model_name)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

        with open(save_json_path, 'w', encoding='utf-8') as fw:
            json.dump(data_dict, fw, indent=4, ensure_ascii=False)
        break
    # break



save_folder_path = os.path.join('../../dataset/final_dataset/generate_block_result_data2024_5_14v2', model_name)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)




