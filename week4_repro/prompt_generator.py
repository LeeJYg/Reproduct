import os
import json
import random   

import sys

from datasets import load_dataset
from tqdm import tqdm

def prompt_concat(question_text, demonstration, mode, separator='\n\n', instance_separator='\n\n\n'):
    prompt = ""
    if mode == 'recitation':
        for question, answer in zip(question_text, demonstration):
            prompt += "Question: " + question + separator
            prompt += "The answer to the above question can be found in the following Wikipedia page, section, and paragraph or table: " + separator
            prompt += "Answer: " + answer + instance_separator
    elif mode == 'answer':
        for question, answer in zip(question_text, demonstration):
            prompt += "Question: " + question + separator
            #prompt += "The answer to the above question can be found in the following Wikipedia page, section, and paragraph or table: " + separator
            prompt += "Answer: " + ' '.join(answer) + instance_separator
        prompt += "Based on the above paragraph, could you answer the following (probably) relevant questions?"
            
    return prompt

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        examples = [json.loads(jline) for jline in f]
    return examples

def print_keys(dictionary):
    for key, value in dictionary.items():
        print(key)
        if isinstance(value, dict):
            print("---------------------")        
            print("isistance: ")
            print_keys(value)
            print("---------------------")  
        elif isinstance(value, list):     
            print("list")
            print("---------------------") 

def prompt_geneartor(mode, shot=2, data_dir='/mnt/sdc/jylee'):
    dataset = load_dataset('natural_questions', cache_dir=data_dir, trust_remote_code=True)
    
    training_data = dataset['train']
    
    selected_data = []
    counter = 0
    
    while len(selected_data) <= shot:
        temp_q = random.choice(training_data)

        if mode == 'recitation':
            if temp_q['annotations']['long_answer'][0]['start_token'] == -1:
                continue  
        elif mode == 'answer':
            if not temp_q['annotations']['short_answers'] or temp_q['annotations']['short_answers'][0]['start_token'] == []:
                continue  

        selected_data.append(temp_q)
        counter += 1  
        
        if counter >= shot:
            break  
    
    question = []
    demonstration = []

    for iter, temp_q in enumerate(selected_data):
        question_text = temp_q['question']['text']
        question.append(question_text)
        #print(len(temp_q['document']['tokens']['token']))
        #print(temp_q['annotations']['long_answer'])
        #print(temp_q['annotations']['short_answers'])

        if mode == 'recitation' and temp_q['annotations']['long_answer']:
            long_answer = temp_q['annotations']['long_answer'][0]
            long_answer_text = " ".join([temp_q['document']['tokens']['token'][i] for i in range(long_answer['start_token'], long_answer['end_token']) if not temp_q['document']['tokens']['is_html'][i]])
            demonstration.append(long_answer_text)
        else:
            long_answer_text = "No long answer provided."

        if mode == 'answer' and temp_q['annotations']['short_answers']:
            short_answer_texts = [" ".join([temp_q['document']['tokens']['token'][i] for i in range(sa['start_token'][0], sa['end_token'][0]) if not temp_q['document']['tokens']['is_html'][i]]) for sa in temp_q['annotations']['short_answers']]
            demonstration.append(short_answer_texts)
        else:
            short_answer_texts = ["No short answers provided."]

    return prompt_concat(question, demonstration, mode)
        
if __name__ == '__main__':
    prompt_geneartor('recitation', 5)