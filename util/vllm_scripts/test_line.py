"""
test line data
"""

import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "/DATA/baseModels/starcoder2-15b"

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
    prompt = f'''
            You will act as a professional Python programming engineer, and I will provide a code snippet where a certain line in the code will be masked and represented as<mask>.
            I will provide a functional description related to this code segment, the dependency packages related to this line of code, and the versions of the dependency packages.
            You need to infer the masked line of code based on this information. Note that you only need to return one line of code, and the line is the response you infer.
            Please be careful not to return the information I provided, only the content of the response needs to be returned Enclose that line of code with tags <start> and <end>. Here is an example:

            ###code snippet：
            for output in outputs:
                prompt = output.prompt
                <mask>
                print("Prompt,Generated text")
            ###Function Description：
            The function of this code is to print the results predicted by calling the model using vllm.
            ###dependeny and version：
            vllm==0.3.3
            ###response:
            <start>generated_text = output.outputs[0].text<end>

            ###code snippet：
            {masked_code}
            ###Function Description：
            {description}
            ###dependeny and version:
            {version}
            ###response:

        '''
    return prompt


json_path = '../dataset/final_dataset/random_line_data_2024_5_14v2/random_line_data_2024_5_14v2.json'

with open(json_path, 'r', encoding='utf-8')as fr:
    lodict = json.load(fr)
data_dict = lodict
data_list = data_dict['data']
test_list = []


for data in data_list:
    version = data['dependency'] + data['version']  #   package == x.x.x
    description = data['description']   #   function description
    masked_code = data['masked_code']   #   masked code

    instruction = bulid_prompt(version, description, masked_code)
    encoded_inputs = tokenizer(instruction, return_tensors="pt")
    if encoded_inputs.input_ids.size(1) > max_length:
        encoded_inputs = {key: tensor[:, :max_length] for key, tensor in encoded_inputs.items()}

    instruction = tokenizer.decode(encoded_inputs['input_ids'][0], skip_special_tokens=True)

    test_list.append(instruction)


sampling_params = SamplingParams(n=6, temperature=0.8, top_p=0.95, max_tokens=128)

llm = LLM(model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.7, swap_space=20)

outputs = llm.generate(test_list, sampling_params)

for output in outputs:
    requests_id = int(output.request_id)
    temp_ans_list = []
    output_list = output.outputs
    for o in output_list:
        text = o.text
        temp_ans_list.append(text)

    data_list[requests_id]['model_output'] = str(temp_ans_list)


save_folder_path = os.path.join('../dataset/generate_line_result_data2024_5_14v2', model_name.split('/')[-1])
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

save_json_path = os.path.join(save_folder_path, json_path.split('/')[-1])

with open(save_json_path, 'w', encoding='utf-8')as fw:
    json.dump(data_dict, fw, indent=4, ensure_ascii=False)
