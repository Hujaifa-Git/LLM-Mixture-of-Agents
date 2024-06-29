import os
import testing_config as ctg
import torch
import json

os.environ['TRANSFORMERS_CACHE'] = ctg.cache_path
os.environ['HF_HOME'] = ctg.cache_path
os.environ['HF_DATASETS_CACHE'] = ctg.cache_path
os.environ['TORCH_HOME'] = ctg.cache_path
os.environ['HF_TOKEN'] = ctg.token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ctg.token

from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ids = ctg.agg_models

queries = ctg.lines
message = [{'role':'system','content':ctg.sys_prompt},
           {'role':'user', 'content':''},
           {'role':'system', 'content':''}]
output_data = {itm:{} for itm in ctg.agg_queries}

for model_id in model_ids:
    print(f'\n\n\n\nIterating:::: {model_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto',attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    for i in range(len(ctg.agg_queries)):
        message[1]['content']=ctg.agg_queries[i]
        message[2]['content']=ctg.agg_responses[i]
    
        input_ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
        input_length = input_ids.shape[1]       
        outputs = model.generate(input_ids,
            max_new_tokens=ctg.agg_max_new_tokens, 
            do_sample=ctg.agg_do_sample,
            temperature=ctg.agg_temperature,#0.7
            top_p=ctg.agg_top_p, #0.75
            top_k=ctg.agg_top_k,
            num_beams = ctg.agg_num_beams,
            early_stopping=ctg.agg_early_stopping, ##
            no_repeat_ngram_size=ctg.agg_no_repeat_ngram_size,
        )
        result = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)[0]
        
        print(f'RESULT: {result}')
        print('*'*20)
        output_data[ctg.agg_queries[i]][model_id] = result
filename = ctg.agg_output_json_dir
with open(filename, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, indent=4, ensure_ascii=False)
