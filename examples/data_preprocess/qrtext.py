import os
import argparse
import json
import pandas as pd
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs

def make_prefix(question, data_description, template_type):
    if data_description:
        question = f"{data_description} {question}"
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> $x = \\boxed{{0.28}}$. </answer>. Or if it is multiple choice, <answer> The correct answer is $\\boxed{{B}}$.</answer>
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> $x = \\boxed{{0.28}}$. </answer>. Or if it is multiple choice, <answer> The correct answer is $\\boxed{{B}}$.</answer><|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

def load_qrtext_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def extract_categories(data):
    categories = {}
    for example in data:
        category = example["meta_data"].get("question_type", "unknown")
        if category not in categories:
            categories[category] = []
        categories[category].append(example)
    return categories

def sample_data(categories, train_size=200):
    train_data, test_data = [], []
    num_categories = len(categories)
    per_category = train_size // num_categories
    
    for category, examples in categories.items():
        examples = examples[:]
        train_data.extend(examples[:per_category])
        test_data.extend(examples[per_category:])
    
    return train_data, test_data

def process_data(data, template_type, split):
    processed_data = []
    for idx, example in enumerate(data):
        question = example["question"]
        data_description = example["data_description"]
        formatted_question = make_prefix(question, data_description, template_type)
        answer = example["answer"]
        
        entry = {
            "data_source": "QRText",
            "prompt": [{"role": "user", "content": formatted_question}],
            "ability": "data_science",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "index": idx,
                "split": split,
                "reference": example["meta_data"].get("reference", ""),
                "keywords": example["meta_data"].get("keywords", []),
                "question_type": example["meta_data"].get("question_type", ""),
                "multiple_choices": example["meta_data"].get("multiple_choices", [])
            }
        }
        processed_data.append(entry)
    return processed_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='/home/hanboli/projects/QRData/benchmark/QRText.json')
    parser.add_argument('--local_dir', default='~/data/qrtext')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    
    args = parser.parse_args()
    
    raw_data = load_qrtext_data(args.file_path)
    categories = extract_categories(raw_data)
    train_data, test_data = sample_data(categories)
    
    train_processed = process_data(train_data, args.template_type, 'train')
    test_processed = process_data(test_data, args.template_type, 'test')
    
    # Print out one example from the training set
    print(json.dumps(train_processed[0], indent=2))
    
    train_df = pd.DataFrame(train_processed)
    test_df = pd.DataFrame(test_processed)
    
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
