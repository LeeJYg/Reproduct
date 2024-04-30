import sys

from prompt_generator import prompt_geneartor
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset

import torch

torch.cuda.empty_cache()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

separator=';;'
instance_separator=';;;'

#tokenizer = AutoTokenizer.from_pretrained("google/ul2")
#model = AutoModel.from_pretrained("google/ul2", device_map="auto")
#model = T5ForConditionalGeneration.from_pretrained("google/ul2", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)                                                                                                  
#dataset = load_dataset('natural_questions')

def inference_iter(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generate_ids = model.generate(inputs.input_ids, max_length=2048, temperature=0.7, do_sample=True)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return output

def get_majority_voting(multiple_outputs, post_process_fn=None):
  majority_voting = {}
  for ans in multiple_outputs:
    if post_process_fn is not None:
      ans = post_process_fn(ans)
    if ans not in majority_voting:
      majority_voting[ans] = 1
    else:
      majority_voting[ans] += 1

  best_ans = list(majority_voting.keys())[0]

  for ans in majority_voting:
    if not ans:
      continue
    if (majority_voting[ans] > majority_voting[best_ans]) or not best_ans:
      best_ans = ans
    if majority_voting[ans] == majority_voting[best_ans] and len(ans) < len(best_ans):
      best_ans = ans

  return best_ans

if __name__ == "__main__":
  tokenizer = AutoTokenizer.from_pretrained("google/ul2")
  #model = AutoModel.from_pretrained("google/ul2", device_map="auto")
  model = T5ForConditionalGeneration.from_pretrained("google/ul2", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)                                                                                                  
  dataset = load_dataset('natural_questions')
  mode_recite = 'recitation'
  mode_answer = 'answer'
  recitation_shot = 5
  answer_shot = 5
  
  self_consistency = 20
  
  total_data_num = len(dataset['valid'])
  
  for iter, data in enumerate(dataset['validation']):
    question = data['question']['text']
    answers = [" ".join([data['document']['tokens']['token'][i] for i in range(sa['start_token'][0], sa['end_token'][0]) if not data['document']['tokens']['is_html'][i]]) for sa in data['annotations']['short_answers']]
    
    prompt_recitation = prompt_geneartor(mode_recite, recitation_shot)
    prompt_recitation += "Question: " + question + separator
    prompt_recitation += "The answer to the above question can be found in the following Wikipedia page, section, and paragraph or table: " + separator
    prompt_recitation += "Answer: "

    generated_text = []
    
    #print(prompt_recitation)
    
    for i in range(self_consistency):
        generated_text.append(inference_iter(prompt_recitation, tokenizer, model, device))
    
    outputs = []
    for text in generated_text:
        #print("Generated Answer: ", text)
        prompt_answer = prompt_geneartor(mode_answer, answer_shot)
        prompt_answer = "Recitation: " + text + separator + "Based on the above paragraph, could you answer the following (probably) relevant questions?" + separator + "Question: " + question + separator
        prompt_answer += "Therefore, the short answer is "
        outputs.append(inference_iter(prompt_answer, tokenizer, model, device))
    
    final_output = get_majority_voting(outputs)
    
    print("Itereation: ", iter, final_output)
    sys.exit()
