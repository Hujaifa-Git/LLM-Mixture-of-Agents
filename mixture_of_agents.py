import os
import config as ctg
os.environ['TRANSFORMERS_CACHE'] = ctg.cache_path
os.environ['HF_HOME'] = ctg.cache_path
os.environ['HF_DATASETS_CACHE'] = ctg.cache_path
os.environ['TORCH_HOME'] = ctg.cache_path
os.environ['HF_TOKEN'] = ctg.token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ctg.token
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_proposer(model_ids, queries):
    prompt = [[{'role':'system', 'content':ctg.sys_initial_prompt}, {'role':'user', 'content':query}] for query in queries]
    respose = []
    for model_id in model_ids:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto',attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        if model_id == 'microsoft/Phi-3-mini-128k-instruct' : tokenizer.chat_template = ctg.phi_chat_template
        
        input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, padding=True,return_tensors='pt').to(device)
        input_length = input_ids.shape[1]       
        outputs = model.generate(input_ids,
            max_new_tokens=ctg.layer1_max_new_tokens,
            num_beams = ctg.layer1_num_beams,
            early_stopping = ctg.layer1_early_stopping, 
            no_repeat_ngram_size = ctg.layer1_no_repeat_ngram_size,
            num_return_sequences = ctg.layer1_num_return_sequences,
            do_sample = ctg.layer1_do_sample,
            top_p = ctg.layer1_top_p,
            top_k = ctg.layer1_top_k,
            temperature = ctg.layer1_temperature            
        )
        result = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        respose.append(result)
    return respose

def generate_aggregator(model_ids, query, previous_responses, final_layer=False):
    agg_prompt = []
    for i in range(len(query)):
        single_query_agg_prompt = ctg.agg_sys_prompt
        for j in range(len(previous_responses)):
            single_query_agg_prompt+=f'Model No.{j+1}s Output :\n {previous_responses[j][i]}\n\n'
        agg_prompt.append(single_query_agg_prompt)
    prompt = [[{'role':'system', 'content':ctg.final_sys_initial_prompt if final_layer else ctg.sys_initial_prompt}, {'role':'user', 'content':single_query}, {'role':'system', 'content':single_agg_prompt}] for single_query, single_agg_prompt in zip(query,agg_prompt)]
    # print(prompt)
    respose = []
    for model_id in model_ids:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto',attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        if model_id == 'microsoft/Phi-3-mini-128k-instruct' : tokenizer.chat_template = ctg.phi_chat_template

        input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, padding=True,return_tensors='pt').to(device)
        input_length = input_ids.shape[1]       
        outputs = model.generate(input_ids,
            max_new_tokens=ctg.final_max_new_tokens if final_layer else ctg.layer2_max_new_tokens,
            num_beams = ctg.final_num_beams if final_layer else ctg.layer2_num_beams,
            early_stopping = ctg.final_early_stopping if final_layer else ctg.layer2_early_stopping, 
            no_repeat_ngram_size = ctg.final_no_repeat_ngram_size if final_layer else ctg.layer2_no_repeat_ngram_size,
            num_return_sequences = ctg.final_num_return_sequences if final_layer else ctg.layer2_num_return_sequences,
            do_sample = ctg.final_do_sample if final_layer else ctg.layer2_do_sample,
            top_p = ctg.final_top_p if final_layer else ctg.layer2_top_p,
            top_k = ctg.final_top_k if final_layer else ctg.layer2_top_k,
            temperature = ctg.final_temperature if final_layer else ctg.layer2_temperature            
        )
        result = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        respose.append(result)
    return respose

if __name__ == '__main__':
    start_time = time.time()
    queries = ctg.user_query
    # print(queries)
    layer1_responses = generate_proposer(ctg.layer1_model_ids,queries)
    print('Layer 1 Responses:')
    print(layer1_responses)
    layer2_responses = generate_aggregator(ctg.layer2_model_ids, queries, layer1_responses, False)
    print('Layer 2 Responses:')
    print(layer2_responses)
    final_response = generate_aggregator(ctg.final_model_id,queries,layer2_responses, True)
    for i in range(len(queries)):
        print(f'QUERY:::\n{queries[i]}\n\nANSWER:::\n{final_response[0][i]}\n\n\n\n\n')
    print(f'Total Time : {time.time() - start_time}')