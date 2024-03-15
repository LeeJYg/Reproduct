from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

#from contriever.src.contriever import Contriever

import jsonlines
import random
import sys
import torch
import spacy
import time

special_tokens = {
  "</paragraph>": 32006,
  "<pad>": 32015,
  "<paragraph>": 32005,
  "[Continue to Use Evidence]": 32002,
  "[Fully supported]": 32012,
  "[Irrelevant]": 32003,
  "[No Retrieval]": 32000,
  "[No support / Contradictory]": 32014,
  "[Partially supported]": 32013,
  "[Relevant]": 32004,
  "[Retrieval]": 32001,
  "[Utility:1]": 32007,
  "[Utility:2]": 32008,
  "[Utility:3]": 32009,
  "[Utility:4]": 32010,
  "[Utility:5]": 32011
}

n_docs = 3
jsonl_file_path = '/home/jylee/Data/triviaqa_test_w_gs.jsonl'
#https://drive.google.com/file/d/1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb/view

model_name = "selfrag/selfrag_llama2_7b"

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

sampling_params = {
    "do_sample": False,
    "top_p": 1.0,
    "max_length": 1024,
}

def format_prompt(input, paragraph=None):
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
    if paragraph is not None:
        prompt += "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(paragraph["title"], paragraph["text"])
    return prompt

def probs_to_token_id(probs):
    token_id = []
    for prob in probs:
        token_id.append(torch.argmax(prob).item())
    return token_id

def seq_prob_sum(probs):
    ret = 0
    for prob in probs:
        ret += torch.log(torch.max(prob))
    return ret

def find_related_passages(query, jsonl_file_path, top_k=1, timeout=30):
    query_keywords = query.lower().split()
    related_passages = []
    start_time = time.time()
    with jsonlines.open(jsonl_file_path) as reader:
        for obj in reader:
            if time.time() - start_time > timeout:
                break

            text = obj["text"].lower()
            title = obj.get("title", "").lower()
            score = sum(text.count(keyword) for keyword in query_keywords)  # 키워드 등장 빈도 점수

            if score > 0:
                related_passages.append((title, obj["text"], score))

    related_passages.sort(key=lambda x: x[2], reverse=True)
    return related_passages[:top_k]


def inference(w_re1=1, w_sup=1, w_use=0.5):
    retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in retrieval_tokens_names}
    
    utility_tokens_names = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    use_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in utility_tokens_names}
    
    support_tokens_names = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
    sup_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in support_tokens_names}
    
    rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
    rel_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in rel_tokens_names}
    
    
    data = []
    with jsonlines.open(jsonl_file_path) as reader:
        for text in reader:
           data.append(text) 
    
    random_element = random.choice(data)      
    
    query = random_element['question']
    evidence = random_element['ctxs'] 
    for evi in evidence:
        if len(evi['text']) > sampling_params["max_length"]:
            evi['text'] = evi['text'][:sampling_params["max_length"]]
    
    input_prompt = format_prompt(query)
    prompt = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
    output = model.generate(prompt, **sampling_params, output_scores=True, return_dict_in_generate=True)

    probs = [torch.softmax(score, dim=-1) for score in output.scores]

    threshold = 0.2
    
    score_dict = {}
    for token, id in ret_tokens.items():
        if id not in probs[0]:
            score_dict[token] = -100
        prob = probs[0][0][id]
        score_dict[token] = float(prob)
    do_retrieve= score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
    
    if do_retrieve:
        augmented_prompt = [format_prompt(query, paragraph=evi) for evi in evidence]
        outputs = []
        for prompt in augmented_prompt:
            prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs.append(model.generate(prompt, **sampling_params, output_scores=True, return_dict_in_generate=True))
        
        relevance_score_dict = {}
        sup_score_dict = {}
        use_score_dict = {}
        final_score_dict = {}
        
        for output_index, output in enumerate(outputs):
            
            relevance_score_dict.setdefault(output_index, {})
            sup_score_dict.setdefault(output_index, {})
            use_score_dict.setdefault(output_index, {})
            
            probs = [torch.softmax(score, dim=-1) for score in output.scores]
            token_id = probs_to_token_id(probs)
            seq_score = seq_prob_sum(probs)
            
            for token, id in rel_tokens.items():
                if id not in probs[0]:
                    prob = torch.exp(torch.tensor(-100.0))
                else:
                    prob = probs[0][0][id]
                relevance_score_dict[output_index][token] = float(prob)
            relevance_score = relevance_score_dict[output_index]["[Relevant]"] / (relevance_score_dict[output_index]["[Relevant]"] + relevance_score_dict[output_index]["[Irrelevant]"])
            
            sup_token_idx = []
            for tok_idx, tok in enumerate(token_id):
                if tok in sup_tokens.values():
                    sup_token_idx.append(tok_idx)
                    break
            if len(sup_token_idx) > 0:
                idx = sup_token_idx[0]
                for token, id in sup_tokens.items():
                    prob = probs[idx][0][id]
                    sup_score_dict[output_index][token] = float(prob)
                    
            if len(sup_score_dict[output_index]) == 3:
                total_sup_prob = (sup_score_dict[output_index]["[Fully supported]"] + sup_score_dict[output_index]["[Partially supported]"] + sup_score_dict[output_index]["[No support / Contradictory]"])
                sup_score = (sup_score_dict[output_index]["[Fully supported]"] / total_sup_prob) + 0.5 * (sup_score_dict[output_index]["[Partially supported]"] / total_sup_prob)
            else:
                sup_score = 0.0
            
            use_token_idx = []
            for tok_idx, tok in enumerate(token_id):
                if tok in use_tokens.values():
                    use_token_idx.append(tok_idx)
                    break
            if len(use_token_idx) > 0:
                idx = use_token_idx[0]
                for token, id in use_tokens.items():
                    prob = probs[idx][0][id]
                    use_score_dict[output_index][token] = float(prob)
                    
            if len(use_score_dict[output_index]) == 5:
                total_use_prob = (use_score_dict[output_index]["[Utility:1]"] + use_score_dict[output_index]["[Utility:2]"] + use_score_dict[output_index]["[Utility:3]"] + use_score_dict[output_index]["[Utility:4]"] + use_score_dict[output_index]["[Utility:5]"])
                weight = [-1, -0.5, 0, 0.5, 1]
                use_score = 0
                for i in range(5):
                    use_score += weight[i] * use_score_dict[output_index]["[Utility:{}]".format(i+1)] / total_use_prob
            else:
                use_score = 0.0    
            
            final_score_dict[output_index] = torch.exp(seq_score) + w_re1 * relevance_score + w_sup * sup_score + w_use * use_score

        max_index = max(final_score_dict, key=final_score_dict.get)
        output = outputs[max_index]

        print(tokenizer.decode(output.sequences[0], skip_special_tokens=False))
     
    else:
        prompt = tokenizer.encode(input_prompt + "[No Retrieval]", return_tensors="pt").to(device)
        output = model.generate(prompt, **sampling_params, output_scores=True, return_dict_in_generate=True)

        print(tokenizer.decode(output.sequences[0], skip_special_tokens=False))
        
if __name__ == "__main__":
    inference()