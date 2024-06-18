"""
GPT performs token level generation prediction and truncates overly long tokens,test c# java javascript,Manually changing data and prompts
"""
import json
import openai
from openai import OpenAI
import os
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_tokens = 127000   #gpt3.5 is 16ktoken    gpt4o is 128k
model_name = "gpt-4o"

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
            You are a professional C# engineer, and I will provide functional descriptions and versions of specified dependency packages. 
            You need to write code in C# to implement this feature based on the functional description and using the dependency package and version I specified. 
            Please note that you only need to return the code that implements the function, and do not return any other content. 
            Please use <start> and <end> to enclose the generated code. Here is an example:
            ###Function Description：
            This code demonstrates how to use the XYZ library (version 2.0) within a C# program. It includes importing the library, fetching its version, and performing various operations as required by your application.
            ###dependeny and version：
            XYZ==2.0
            ###response:
            <start>
            using ThirdPartyLibrary; 

            class Program
            {{
                static void Main(string[] args)
                {{

                    string libraryVersion = ThirdPartyLibrary.Version;


                }}
            }}
            <end>

            ###Function Description：
            {description}
            ###dependeny and version：
            {version}
            ###response:


        '''
    return prompt



json_path = '../../dataset/final_dataset/other_language_data_finalv2/java_test_block.json'
# json_path = '../../dataset/final_dataset/other_language_data_finalv2_result/gpt-3.5-turbo/java_test_block.json'  #After an error occurs, use the address to continue predicting from where the error occurred


with open(json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']



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
        save_folder_path = os.path.join('../../dataset/final_dataset/other_language_data_finalv2_result', model_name)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

        with open(save_json_path, 'w', encoding='utf-8') as fw:
            json.dump(data_dict, fw, indent=4, ensure_ascii=False)
        break
    # break



save_folder_path = os.path.join('../../dataset/final_dataset/other_language_data_finalv2_result', model_name)
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)




