import sys
import openai

from prompt_generator import prompt_geneartor
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference_iter(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=2048, no_repeat_ngram_size=2)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    output_text = output[len(prompt):]
    output_text_index = output_text.find('\n\n')
    output_text = output_text[:output_text_index].strip()

    return output_text


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
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-30b")
    
    mode_recite = 'recitation'
    mode_answer = 'answer'
    recitation_shot = 5
    answer_shot = 64
    
    self_consistency = 3
    
    question = "what does the word china mean in chinese?"    
    
    prompt_recitation = prompt_geneartor(mode_recite, recitation_shot)
    prompt_recitation += "Question: " + question + '\n\n'
    prompt_recitation += "The answer to the above question can be found in the following Wikipedia page, section, and paragraph or table: \n\n"
    prompt_recitation += "Answer: "

    generated_text = []
    
    print(prompt_recitation)
    
    for i in range(self_consistency):
        generated_text.append(inference_iter(prompt_recitation, tokenizer, model))
    
    outputs = []
    for text in generated_text:
        print("Generated Answer: ", text)
        prompt_answer = text + '\n\n' + "Based on the above paragraph, could you answer the following (probably) relevant questions?\n\n" + "Question: " + question + '\n\n'
        prompt_answer += "Answer: "
        outputs.append(inference_iter(prompt_answer, tokenizer, model))
    
    final_output = get_majority_voting(outputs)
    
    print(final_output)
