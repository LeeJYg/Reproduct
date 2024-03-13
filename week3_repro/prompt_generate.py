import os
import json
from typing import List, Dict

import sys
import _jsonnet
import random

def read_json(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        instance = json.load(file)
    return instance

def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances

def prompt_generte(purpose:str, distractor:int):
    print("Prompt generation is in progress...")
    
    #propossed_data/hotpotqa에서 purpose에 맞는 파일 읽어들이기.
    if purpose == "train":
        file_path = os.path.join("processed_data", "hotpotqa", "train.jsonl")
        content = read_jsonl(file_path)
    else:
        file_path = os.path.join("processed_data", "hotpotqa", "dev.jsonl")
        content = read_jsonl(file_path)
    
    annotated_data_file_path = os.path.join("annotated_data", "hotpotqa.jsonnet")
    annotations = json.loads(_jsonnet.evaluate_file(annotated_data_file_path))
    
    generated_prompt = []
    
    for instance in annotations:
        id = instance['question_id']
        processed_data = None
        for i in content:
            if i['question_id'] == id:
                processed_data = i
                break
        if processed_data == None:
            print("ERROR: Couldn't find any match for the annotated paragraph.")
            continue
        
        #processed_data['context']에서 is_supporting = true인 애들만 뽑기
        context = processed_data['contexts']
        supporting_context = []
        for i in context:
            if i['is_supporting'] == True:
                supporting_context.append(i)
        
        non_supporting_idx = [i['idx'] for i in context if i not in supporting_context]
        distractor_idx = random.sample(non_supporting_idx, distractor)
        distractor_context = []
        for i in context:
            if i['idx'] in distractor_idx:
                distractor_context.append(i)
            
        prompt = ""
        prompt += "# METADATA: {" + "qid: " + str(id) + "}" + "\n"
            
        #paragraph 순서 무작위로 섞기
        paragraph = supporting_context + distractor_context
        random.shuffle(paragraph)
        
        for inst in paragraph:
            prompt += "Wikipedia Title: " + inst['title'] + "\n"
            prompt += inst["paragraph_text"] + "\n\n"
        
        prompt += "Q: " + processed_data['question_text'] + "\n"
        
        cot = ""
        for reasoning in instance["reasoning_steps"]:
            cot += reasoning['cot_sent'] + ' '
        
        prompt += "A: " + cot + "\n"
        generated_prompt.append(prompt)
    
    directory = os.path.join("prompt", "hotpotqa")
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    #directory에 generated_prompt를 저장. txt 파일로.
    file_path = os.path.join(directory, purpose + ".txt")
    with open(file_path, 'w') as f:
        for prompt in generated_prompt:
            f.write(prompt)
            f.write('\n\n')
    
    print("Prompt generation is complete!")
    
if __name__ == "__main__":
    prompt_generte("dev", 1)