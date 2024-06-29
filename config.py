user_query = ['Why did the dinosaures got extinct?', 
              'What is the biggest country in the world?', 
              'Where is America located?', 
              'What is a samurai?', 
            ]

layer1_model_ids = ['Qwen/Qwen2-1.5B-Instruct', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0']
# layer1_model_ids = ['Qwen/Qwen2-1.5B-Instruct', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'microsoft/Phi-3-mini-128k-instruct','stabilityai/stablelm-zephyr-3b']
layer2_model_ids = ['microsoft/Phi-3-mini-128k-instruct','stabilityai/stablelm-zephyr-3b']
final_model_id = ['/home/nsl3090-3/.cache/huggingface/hub/models--teknium--OpenHermes-2.5-Mistral-7B/snapshots/24c0bea14d53e6f67f1fbe2eca5bfe7cae389b33']

layer1_max_new_tokens=128
layer1_num_beams = 5
layer1_early_stopping = True
layer1_no_repeat_ngram_size=5
layer1_num_return_sequences=1
layer1_do_sample = True
layer1_top_k = 20#50
layer1_top_p = 0.9#0.8 
layer1_temperature = 0.3#0.8


layer2_max_new_tokens=128
layer2_num_beams = 5
layer2_early_stopping = True
layer2_no_repeat_ngram_size=5
layer2_num_return_sequences=1
layer2_do_sample = True
layer2_top_k = 20
layer2_top_p = 0.9 
layer2_temperature = 0.2


final_max_new_tokens=256
final_num_beams = 5
final_early_stopping = True
final_no_repeat_ngram_size=5
final_num_return_sequences=1
final_do_sample = True
final_top_k = 10
final_top_p = 0.95 
final_temperature = 0.1

cache_path = '/media/nsl3090-3/hdd1/Huggingface_Cache'
token = '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'

sys_initial_prompt = "You are an AI assistant. Please provide a helpful and accurate response to the following user query. Keep your response as short and compact as possible." ## more refine
final_sys_initial_prompt = "You are an AI assistant. Please provide a helpful and accurate response to the following user query."
agg_sys_prompt = """You are a helpful chat assistant.You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
"""

phi_chat_template = """{{ bos_token }}
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>' + '\n' + message['content'] + '<|end|>' }}
{% elif message['role'] == 'system' %}
{{ '<|system|>' + '\n' + message['content'] + '<|end|>' }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>' + '\n' + message['content'] + '<|end|>' }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{'
' + '<|assistant|>' + '
' }}
{% endif %}
{% endfor %}"""
