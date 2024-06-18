"""
test c# data
"""

import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_name = "/DATA/baseModels/CodeGemma-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
max_length = model.config.max_position_embeddings  # The maximum context length of the model
print(f"max_length:{max_length}")

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


json_path = '../dataset/final_dataset/new_c#_data/c#_test_block.json'

with open(json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']
test_list = []


for data in data_list:
    version = data['dependency'] + data['version']  #   package == x.x.x
    description = data['description']   #   function description

    instruction = bulid_prompt(version, description)
    encoded_inputs = tokenizer(instruction, return_tensors="pt")
    if encoded_inputs.input_ids.size(1) > max_length:
        encoded_inputs = {key: tensor[:, :max_length] for key, tensor in encoded_inputs.items()}

    test_list.append(instruction)


sampling_params = SamplingParams(n=6, temperature=0.8, top_p=0.95, max_tokens=512)

llm = LLM(model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.7, swap_space=40)


outputs = llm.generate(test_list, sampling_params)

for output in outputs:
    requests_id = int(output.request_id)
    temp_ans_list = []
    output_list = output.outputs
    for o in output_list:
        text = o.text
        temp_ans_list.append(text)

    data_list[requests_id]['model_output'] = str(temp_ans_list)


save_folder_path = os.path.join('../dataset/new_c#_data_result', model_name.split('/')[-1])
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)




