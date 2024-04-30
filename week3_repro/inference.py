import os
import sys

from nltk import sent_tokenize
from elasticsearch_test import retrieval
from transformers import T5ForConditionalGeneration, T5Tokenizer
from prompt_generate import read_jsonl

import re
import torch
import random
from collections import deque

import nltk

nltk.download('punkt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16, max_length=2048)

model = model.to(device)

def inference_iter(prompt):
    
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generate_ids = model.generate(inputs, max_new_tokens=2048)
        output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        
    #print("Output: ", output)

    torch.cuda.empty_cache()

    return output

def process_queue(queue):
    ret = ""
    for item in queue:
        ret += item
        
    return ret

def clean_and_extract_qids(prompts, n):
    pattern = re.compile(r"# METADATA: {\"qid\": \"(\w+)\"}")
    
    qid_list = []
    cleaned_prompts = []
    split_prompts = prompts.split('\n\n\n')

    for line in split_prompts[:n]:
        lines = line.split('\n')
        for l in lines:
            match = pattern.match(l)
            if match:
                qid_list.append(match.group(1))
            else:
                cleaned_prompts.append(l)
                
    cleaned_prompts_str = '\n'.join(cleaned_prompts)
    
    return cleaned_prompts_str, qid_list

def calculate_f1(predicted, truth):
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(truth.lower().split())
    common_tokens = pred_tokens.intersection(truth_tokens)
    if not common_tokens:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_recall(retrieved_items, supporting_facts):
    retrieved_set = {item['title'] for item in retrieved_items}
    truth_set = {fact['title'] for fact in supporting_facts if fact['is_supporting']}
    if not truth_set:
        return 1.0  # Avoid division by zero
    recall = len(truth_set.intersection(retrieved_set)) / len(truth_set)
    return recall

    
if __name__ == "__main__":
    #prompt_file_path = os.path.join("prompt", "hotpotqa", "train_flant5.txt")
    prompt_file_path = '/home/jylee/Reproduct/week3_repro/prompt/hotpotqa/train_flant5.txt'
    with open(prompt_file_path, 'r') as f:
        prompts = f.read()    
        
    demonstrations, qid_list = clean_and_extract_qids(prompts, n=8)

    dev_file_path = "/home/jylee/Reproduct/week3_repro/processed_data/hotpotqa/dev.jsonl"
    dev_data = read_jsonl(dev_file_path)
    random.shuffle(dev_data)
    
    #test_num = 10

    test_example = random.choice(read_jsonl(dev_file_path))
    num = 0
    dev_examples = []
    #while num < test_num:
    #    test_example = random.choice(read_jsonl(dev_file_path))
    #    if test_example['question_id'] not in qid_list:
    #        dev_examples.append(test_example)
    #        num += 1
    
    f1_scores = []
    nor_f1_scores = []
    recall_scores = []

    for i, test_example in enumerate(dev_data):
        test_question_nor = "Q: Answer the following question.\n" + test_example['question_text']
        test_question = test_example['question_text']
        
        test_corpus = test_example['contexts']
        test_context = []
        for ctx in test_corpus:
            if ctx['is_supporting']:
                test_context.append(ctx)

        test_context_prompt = ""
        for ctx in test_context:
            test_context_prompt += "Wikipedia Title: " + ctx['title'] + "\n" + ctx['paragraph_text'] + "\n\n"
        
        test_answer = "A: " + test_example['answer']

        #print(test_question_nor)
        nor_output = inference_iter(test_question)
        #print(test_answer)
        
        query = demonstrations + '\n\n' + test_question + '\n'
        q_index = len(prompts)
        cot = "" 

        collected_cot = []
        collected_paragraph = deque(maxlen=15)
        
        iter_num = 0
        max_iter = 3
        
        #1st or => and
        while test_example['answer'].lower() not in cot.lower() and iter_num < max_iter:
            if cot != "":
                retrieved_result  = retrieval(test_example, cot)
            else:
                retrieved_result  = retrieval(test_example, test_question)
            
            for ret in retrieved_result:
                retrieved = "Wikipedia Title: " + ret['title'] + "\n" + ret['paragraph_text'] + "\n\n"
                if retrieved not in collected_paragraph:
                    collected_paragraph.append(retrieved) 
                
            retrieved_prompt = process_queue(collected_paragraph)
            query = demonstrations + '\n\n\n' + retrieved_prompt + 'Q: Answer the following question.\n' + test_question + '\nA: ' + cot

            cot = inference_iter(query)
            collected_cot.append(cot)

            iter_num += 1
        
        final_cot = ""
        for c in collected_cot:
            final_cot += c
            
        final_query = process_queue(collected_paragraph) + "Q: Answer the following question.\n" + test_example['question_text']

        #print("Final Query: \n", final_query)
        
        output = inference_iter(final_query)
        
        if "A:" in output:
            answer_index = output.index("A: ")
            answer = output[answer_index + len("A: "):]
            ret = answer
        else:
            ret = output
            
        #print("Answer: ", ret)
        
        ircot_f1 = calculate_f1(ret, test_answer)
        ircot_recall = calculate_recall(retrieved_result, test_example['contexts'])
        
        nor_f1 = calculate_f1(ret, nor_output)

        f1_scores.append(ircot_f1)
        nor_f1_scores.append(nor_f1)
        recall_scores.append(ircot_recall)
        
        if i % 10 == 0:
            print("Iteration: ", i)
            
            average_f1 = sum(f1_scores) / len(f1_scores)
            average_nor_f1 = sum(nor_f1_scores) / len(nor_f1_scores)
            average_recall = sum(recall_scores) / len(recall_scores)
            
            print(f"Average F1 Score: {average_f1:.3f}")
            print(f"Average No Retrieval F1 Score: {average_nor_f1:.3f}")
            print(f"Average Recall: {average_recall:.3f}")
    