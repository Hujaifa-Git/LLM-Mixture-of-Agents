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
model_ids = ctg.proposer_models
queries = ctg.proposer_prompts



output_data = {itm:{} for itm in queries}

for model_id in model_ids:
    print(f'\n\n\n\nIterating:::: {model_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto',attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    
    message = []
    for query in queries:
        message.append({'role':'user', 'content':query})
        input_ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        input_length = input_ids.shape[1]       
        outputs = model.generate(input_ids,
            max_new_tokens=ctg.proposer_max_new_tokens, 
            do_sample=ctg.proposer_do_sample,
            temperature=ctg.proposer_temperature,#0.7
            top_p=ctg.proposer_top_p, #0.75
            top_k=ctg.proposer_top_k,
            early_stopping=ctg.proposer_early_stopping, ##
            no_repeat_ngram_size=ctg.proposer_no_repeat_ngram_size,
        )
        result = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)[0]
        
        print('*'*20)
        print(f'QUERY: {query}')
        print(f'RESULT: {result}')
        print('*'*20)
        
        output_data[query][model_id] = result
        message.pop()
        

filename = ctg.proposer_output_json_dir
with open(filename, 'w') as json_file:
    json.dump(output_data, json_file, indent=4, ensure_ascii=False)