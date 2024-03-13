import os
import openai
import sys

from retrieval import wiki_retrieval
from nltk import sent_tokenize
from prompt_generate import read_jsonl

import json
from collections import deque

import nltk

nltk.download('punkt')

# OpenAI API 키 설정
openai.api_key = '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'

def inference_iter(question):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )

    output = response['choices'][0]['message']['content']
    cot = sent_tokenize(output)[0]

    print(cot)
    return cot

def process_queue(queue):
    ret = ""
    while queue:
        item = queue.popleft()
        ret += item
        
    return ret
    
    
if __name__ == "__main__":
    prompt_file_path = os.path.join("prompt", "hotpotqa", "train.txt")
    with open(prompt_file_path, 'r') as f:
        prompts = f.read()    

    # dev.jsonl 파일 경로
    dev_file_path = "processed_data/hotpotqa/dev.jsonl"

    # dev.jsonl 파일에서 첫 번째 요소 가져오기
    test_example = read_jsonl(dev_file_path)[2]
    test_question = "Q: Answer the following question by reasoning step-by-step. " + test_example['question_text']
    
    print(test_question)
    
    query = prompts + '\n' + test_question + '\n'
    q_index = len(prompts)
    cot = "" 

    collected_cot = []
    collected_paragraph = deque(maxlen=15)
    iter_num = 0
    while "answer is:" not in cot and iter_num < 8:
        if cot != "":
            retrieved_result  = wiki_retrieval(cot)
        else:
            retrieved_result  = wiki_retrieval(test_example['question_text'])

        retrieved_paragraph, retrieved_title = retrieved_result
        
        for paragraph, title in zip(retrieved_paragraph, retrieved_title):
            retrieved = "Wikipedia Title: " + title + "\n" + paragraph + "\n\n"
            collected_paragraph.append(retrieved)  # Add retrieved paragraph to collected_paragraph

        retrieved_prompt = process_queue(collected_paragraph)
        query = query[:q_index] + retrieved_prompt + query[q_index:] + 'A: ' + cot

        cot = inference_iter(query)
        collected_cot.append(cot)

        iter_num += 1
    
    final_cot = ""
    for c in collected_cot:
        final_cot += c
        
    final_query = process_queue(collected_paragraph) + "Q: " + test_example['question_text']
    
    output = inference_iter(final_query)
    
    if "answer is:" in output:
        answer_index = output.index("answer is:")
        answer = output[answer_index + len("answer is:"):]
        ret = answer
    else:
        ret = output
    
    print(ret)