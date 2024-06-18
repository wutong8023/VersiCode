"""
test token data
"""

import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_name = "/root/WWG/Meta-Llama-3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
max_length = model.config.max_position_embeddings  # The maximum context length of the model
print(f"max_length:{max_length}")

def bulid_prompt(version, description, masked_code) -> str:
    """
    construct prompt
    :param version:
    :param description:
    :param masked_code:
    :param options:
    :return:
    """
    prompt = f"""
        You are a professional Python programming engineer, and I will give you a code snippet where function names are masked and represented as<mask>in the code. 
        There may be multiple <mask>, and all the blocked content in these <mask> is the same. 
        I will provide a functional description of this code, the dependency package to which the function belongs and the version of the dependency package.
        What you need to do is infer what the masked function name is based on this information. You only need to return one content, not every<mask>.
        Please note that you only need to return one function name and do not need to return any other redundant content, and the response is enclosed by <start> and <end> Here is an example:
        ###code snippet：
        outputs = llm.<mask>(prompts, sampling_params)
        ###Function Description：
        This code passes prompts and parameters to the model to obtain the output result of the model.
        ###dependeny and version：
        vllm==0.3.3
        ###response：
        <start>generate<end>

        ###code snippet：
        {masked_code}
        ###Function Description：
        {description}
        ###dependeny and version：
        {version}
        ###response：
        """

    return prompt


json_path = '../dataset/final_dataset/random_token_data_2024_5_14v2/random_token_data_2024_5_14v2.json'

with open(json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']
test_list = []

for data in data_list:
    version = data['dependency'] + data['version']  #   package == x.x.x
    description = data['description']   #   function description
    masked_code = data['masked_code']   #   masker code

    instruction = bulid_prompt(version, description, masked_code)
    encoded_inputs = tokenizer(instruction, return_tensors="pt")
    if encoded_inputs.input_ids.size(1) > max_length:
        encoded_inputs = {key: tensor[:, :max_length] for key, tensor in encoded_inputs.items()}

    instruction = tokenizer.decode(encoded_inputs['input_ids'][0], skip_special_tokens=True)

    test_list.append(instruction)


sampling_params = SamplingParams(n=100, temperature=0.8, top_p=0.95, max_tokens=64)

llm = LLM(model=model_name, tensor_parallel_size=2, gpu_memory_utilization=0.9, swap_space=40)

outputs = llm.generate(test_list, sampling_params)

for output in outputs:
    requests_id = int(output.request_id)
    temp_ans_list = []
    output_list = output.outputs
    for o in output_list:
        text = o.text
        temp_ans_list.append(text)

    data_list[requests_id]['model_output'] = str(temp_ans_list)


save_folder_path = os.path.join('../dataset/generate_result_data2024_5_14v2', model_name.split('/')[-1])
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)



