from prompt_generator import prompt_geneartor
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from inference import inference_iter, get_majority_voting

import time
import torch

#torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

separator='\n\n'

tokenizer = AutoTokenizer.from_pretrained("google/ul2")
#model = AutoModel.from_pretrained("google/ul2", device_map="auto")
model = T5ForConditionalGeneration.from_pretrained("google/ul2", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)          

model = model.to(device)

dataset = load_dataset('natural_questions', split='train')
train_dataset, validation_dataset = dataset.train_test_split(test_size=0.2).values()

def has_short_answers(example):
    return any(sa['text'] for sa in example['annotations']['short_answers'])

filtered_validation_dataset = []
for example in validation_dataset:
    if has_short_answers(example):
        filtered_validation_dataset.append(example)
        if len(filtered_validation_dataset) % 100 == 0:
            print(f"Filtered {len(filtered_validation_dataset)} examples so far.")
    if len(filtered_validation_dataset) == 5000:
        break

start_time = time.time()
print("Check time Start")

mode_recite = 'recitation'
mode_answer = 'answer'
recitation_shot = 5
answer_shot = 5

self_consistency = 20

total_data_num = len(filtered_validation_dataset)

iter = 1
data = filtered_validation_dataset[iter]

question = data['question']['text']
#answers = [" ".join([data['document']['tokens']['token'][i] for i in range(sa['start_token'][0], sa['end_token'][0]) if not data['document']['tokens']['is_html'][i]]) for sa in data['annotations']['short_answers']]
answers = [sa['text'] for sa in data['annotations']['short_answers'] if sa["text"]]

prompt_recitation = prompt_geneartor(mode_recite, recitation_shot, dataset=train_dataset, separator='\n\n', instance_separator='\n\n\n')
prompt_recitation += "Question: " + question + separator
prompt_recitation += "The answer to the above question can be found in the following Wikipedia page, section, and paragraph or table." + separator
prompt_recitation += "Answer: "

print(prompt_recitation)

#S-Denoiser
prompt_recitation = '[NLG] ' + prompt_recitation + ' <extra_id_0>'

generated_text = []

#print(prompt_recitation)

for i in range(self_consistency):
    generated_text.append(inference_iter(prompt_recitation, tokenizer, model, device))
    print("Iter: ", i, ", Generated Recitation: ", generated_text[-1])
    t = time.time()
    print("Time: ", t - start_time)

outputs = []

prompt_answer_base = prompt_geneartor(mode_answer, answer_shot, dataset=train_dataset, separator='\n\n', instance_separator='\n\n\n')

for text in generated_text:
    #print("Generated Answer: ", text)
    prompt_answer = "Recitation: " + text + separator + "Based on the above paragraph, could you answer the following (probably) relevant questions?" + separator + prompt_answer_base + separator + "Question: " + question + separator
    prompt_answer += "Therefore, the short answer is "
    prompt_answer = '[S2S] ' + prompt_answer + ' <extra_id_0>'
    outputs.append(inference_iter(prompt_answer, tokenizer, model, device))
    print("Iter: ", i, ", Generated Answer: ", outputs[-1])
    t = time.time()
    print("Time: ", t - start_time)

final_output = get_majority_voting(outputs)

print("Itereation: ", iter, final_output, answers)

end_time = time.time()

print("Time: ", end_time - start_time)
