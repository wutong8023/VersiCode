"""
test code editing old to
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

def bulid_prompt(description, old_version, old_code, new_version) -> str:
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
    including the dependencies and versions used in the code. Then, I will provide the same dependencies but with a specified new version. 
    Your task is to refactor the code using the methods provided by the specified new version and return the refactored code. 
    Please note that you only need to return the refactored code and enclose it with <start> and <end>:
    ###Functionality description of the code
    {description}
    ###Dependency and old version
    {old_version}
    ###Old version code
    {old_code}
    ###Dependency and new version
    {new_version}
    ###Refactored new code
    """

    return prompt


json_path = '../dataset/final_dataset/code_editing_sample_all_feature/code_editing_sample_all_feature.json'

with open(json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']
test_list = []


for data in data_list:
    old_version = data['dependency'] + data['old_version']  # package == x.x.x
    new_version = data['dependency'] + data['new_version']  # package == x.x.x
    description = data['description']  # function description
    old_code = data['old_code']

    instruction = bulid_prompt(description, old_version, old_code, new_version)

    encoded_inputs = tokenizer(instruction, return_tensors="pt")
    if encoded_inputs.input_ids.size(1) > max_length:
        encoded_inputs = {key: tensor[:, :max_length] for key, tensor in encoded_inputs.items()}

    instruction = tokenizer.decode(encoded_inputs['input_ids'][0], skip_special_tokens=True)

    test_list.append(instruction)


sampling_params = SamplingParams(n=6, temperature=0.8, top_p=0.95, max_tokens=512)

llm = LLM(model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.6, swap_space=40)

outputs = llm.generate(test_list, sampling_params)

for output in outputs:
    requests_id = int(output.request_id)
    temp_ans_list = []
    output_list = output.outputs
    for o in output_list:
        text = o.text
        temp_ans_list.append(text)

    data_list[requests_id]['model_output'] = str(temp_ans_list)


save_folder_path = os.path.join('../dataset/code_editing_sample_all_feature_result', model_name.split('/')[-1])
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)
