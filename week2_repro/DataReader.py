import sys as system
import os
import json
import torch
import transformers
import copy

from torch.utils.data import Dataset
from transformers import Trainer
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import os

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_input": (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_path: Optional[str] = field(default=None)
    use_special_token: bool = field(
        default=False,
        metadata={
            "help": "Use special command during training."
        },
    )
    not_consider_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Consider special tokens during loss calculations."
        },
    )
    use_context_markups: bool = field(
        default=False,
        metadata={
            "help": "make separated training data."
        },
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={
            "help": "use flash attention."
        },
    )
@dataclass    
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    separated: bool = field(
        default=False,
        metadata={
            "help": "make separated training data."
        },
    )
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default='/home/jylee/week2/test/cache')
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    
    print("Data preprocessed successfully!")

    return dict(input_ids=input_ids, labels=labels)

IGNORE_INDEX = -100

class MyDataset(Dataset):
    def __init__(self, sources, targets, tokenizer):
        data_dict = preprocess(sources, targets, tokenizer)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return dict(
            input_ids=self.input_ids[idx],
            labels=self.labels[idx],
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path if model_args.tokenizer_path is None else model_args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        add_special_tokens=True,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "additional_special_tokens": ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]", "[Fully supported]", "[Partially supported]", "[No support / Contradictory]"],
        }
    )
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})    

    dataset_folder = '/mnt/sdc/jylee'
    dataset_files = [file for file in os.listdir(dataset_folder) if file.endswith('.json')]
    dataset_path = os.path.join(dataset_folder, dataset_files[0])

    with open(dataset_path, 'r') as file:
        list_data_dict = json.load(file)
        file.close()

    prompt_input = PROMPT_DICT["prompt_input"]
    sources = [
        prompt_input.format_map(example)
        for example in list_data_dict
    ]
    targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
 
    print("Data loaded successfully!")
    
    dataset = MyDataset(sources, targets, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )
    
    num_epochs = training_args.num_train_epochs
    num_steps = len(dataset) // training_args.per_device_train_batch_size * num_epochs

    print(num_epochs, num_steps)
    
    trainer.train()
    trainer.save_state()

if __name__ == "__main__":
    main()