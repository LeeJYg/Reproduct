from tqdm import tqdm
from datasets import load_dataset

import sys

import os
import json


def write_hotpotqa_instances(instance):

    max_num_tokens = 1000  # clip later.

    processed_instances = {}
    processed_instances["question_id"] = instance['id']
    processed_instances['level'] = instance['level']
    processed_instances["question_text"] = instance['question']
    processed_instances['answer'] = instance['answer']
    
    gold_paragraph_title = list(set(instance['supporting_facts']['title']))
    
    title_to_paragraph = {title: ''.join(text) for title, text in zip(instance['context']['title'], instance['context']['sentences'])}
    paragraph_to_title = {''.join(text): title for title, text in zip(instance['context']['title'], instance['context']['sentences'])}
    
    paragraph_texts = [''.join(text) for text in instance['context']['sentences']]
    
    processed_instances['contexts'] = []
    for index, paragraph in enumerate(paragraph_texts):
        context = {}
        context['idx'] = index
        context['title'] = paragraph_to_title[paragraph]
        context['paragraph_text'] = paragraph[:max_num_tokens]
        context['is_supporting'] = context['title'] in gold_paragraph_title
        processed_instances['contexts'].append(context)
    
    return processed_instances

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("hotpot_qa", "distractor")
    
    directory = os.path.join("processed_data", "hotpotqa")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    train_filepath = os.path.join(directory, "train.jsonl")
    with open(train_filepath, 'w') as f1:
        for instance in tqdm(dataset["train"]):
            f1.write(json.dumps(write_hotpotqa_instances(instance)))
            f1.write('\n')


    dev_filepath = os.path.join(directory, "dev.jsonl")
    with open(dev_filepath, 'w') as f2:
        for instance in tqdm(dataset["validation"]):
            f2.write(json.dumps(write_hotpotqa_instances(instance)))
            f2.write('\n')
