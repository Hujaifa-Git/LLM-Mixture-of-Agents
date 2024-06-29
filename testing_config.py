cache_path = '/media/nsl3090-3/hdd1/Huggingface_Cache'
token = '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'

sys_prompt = 'You are an AI assistant. Please provide a helpful and accurate response to the following user query. Keep your response as short and compact as possible.'

#Proposer Hyper-parameters

proposer_models = ['Qwen/Qwen2-1.5B-Instruct', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'microsoft/Phi-3-mini-128k-instruct', 
            'stabilityai/stablelm-zephyr-3b', 'HuggingFaceH4/zephyr-7b-beta', 'teknium/OpenHermes-2.5-Mistral-7B']

proposer_prompts = ['A farmer has 17 sheep and all but 9 die. How many sheep does the farmer have left?',
           'If you have a 3-gallon jug and a 5-gallon jug, how can you measure exactly 4 gallons of water using these jugs?',
           'Write a Python function that takes a list of numbers and returns the sum of all the even numbers in the list.',
           'Write a Python function to check if a given string is a palindrome (reads the same backward as forward).',
           'Identify the main topic of the following sentence: "The quick brown fox jumps over the lazy dog.',
           'Determine the sentiment (positive, negative, or neutral) of this review: "The product quality is excellent, and I am very satisfied with my purchase.',
           'You have a wolf, a goat, and a cabbage, and you need to cross a river with a boat that can only carry you and one other item at a time. If left together, the wolf will eat the goat, and the goat will eat the cabbage. How do you get everything across safely?',
           'A train leaves Station A at 10:00 AM and travels to Station B at 60 miles per hour. Another train leaves Station B at 11:00 AM and travels to Station A at 90 miles per hour. The distance between Station A and Station B is 450 miles. When do the two trains meet?']

proposer_max_new_tokens=512 
proposer_do_sample=True
proposer_temperature=0.7#0.7
proposer_top_p=0.95 #0.75
proposer_top_k=40
proposer_early_stopping=True ##
proposer_no_repeat_ngram_size=2

proposer_output_json_dir = 'output_v3.json'


#Aggregator Hyper-parameters

agg_models = ['Qwen/Qwen2-1.5B-Instruct', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'microsoft/Phi-3-mini-128k-instruct', 
            'stabilityai/stablelm-zephyr-3b', 'HuggingFaceH4/zephyr-7b-beta', 'teknium/OpenHermes-2.5-Mistral-7B']
agg_queries = ['Complete this story. Once upon a time in a small village, there lived a young girl named Ella who loved adventures. One day, she found a mysterious map in her attic.',
               'Explain how a neural network learns to classify images.',
               'What steps should someone take to become an author?',
               'What are the health benefits of regular exercise?'
               ]

agg_responses = ["""You are a helpful chat assistant.You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
1. Ella decided to follow the map and soon discovered it led to a hidden forest just outside her village. As she ventured deeper, she encountered various magical creatures.
2. Ella ate the map and then decided to go to the bathroom
3. Ella showed the map to her best friend, Tom, and together they prepared for a thrilling journey. They packed their bags with food, water, and a flashlight.
4. Ella was busy that day so she destroyed the map and start studying
5. Ella was unsure what the map led to, but she felt a strange pull towards the forest marked on it. She set off early in the morning, her heart full of excitement and fear.""",
"""You are a helpful chat assistant.You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
1. A neural network learns to classify images by adjusting weights through backpropagation based on the error rate of the output compared to the expected result.
2. During training, a neural network uses labeled data to understand patterns and features within the images. It adjusts its internal parameters to minimize the difference between predicted and actual labels.
3. Neural networks use layers of interconnected nodes to process input images. Through repeated iterations and learning from errors using techniques like gradient descent, the network improves its classification accuracy.
4. During training the neural network stores the image and do similairy match with the target image to classify.
5. Neural networks have magical properties that allows them to classify any kind of images""",
"""You are a helpful chat assistant.You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
1. Begin my watching movies and listening to music. It will make you an author
2. To become an author, you need to drink a lot of alchohol. It makes you more creative
3. To become an author, one should start by developing a love for reading and writing. It's important to choose a niche and understand the publishing industry.
4. Begin by writing regularly and seeking feedback. Research different publishing options, including traditional publishing and self-publishing, to understand the best path for you.
5. Becoming an author involves finishing your work, marketing and promoting it, and continuously improving your writing skills through practice and learning.""",
"""You are a helpful chat assistant.You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
1. Excersice only wastes your precious time
2. There is no point in excersing. learn boxing instead
3. There is no benefit of exercise, it will only make you more and more tired
4. You shouldn't exercise because you may hurt yourself while doing so
5. You should only exercise if you are extremly fat.""",
]

agg_max_new_tokens=256
agg_do_sample=True
agg_temperature=0.5
agg_top_p=0.90
agg_top_k=40
agg_num_beams = 5
agg_early_stopping=True
agg_no_repeat_ngram_size=5

agg_output_json_dir = 'output_agg_v1.json'
