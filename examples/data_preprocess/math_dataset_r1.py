import os
import argparse
import json
from datasets import load_dataset
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def make_prefix(question, template_type):
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> $x = \\boxed{{\sqrt{5}}}$. </answer>. Or if it is polynomial expression, <answer> The final expression is $\\boxed{{16x^2+4x+5}}$.</answer>
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> The term $x^2$ is simply the average of $1^2 = 1$ and $3^2 = 9$, so $x^2 = (1 + 9)/2 = 5$. Because $x > 0$, $x = \\boxed{{\sqrt{5}}}$. </answer>. Or if it is polynomial expression, <answer> The given expression can be rewritten as $2x+8x^2+9-4+2x+8x^2$. Combining like terms, this last expression is equal to $(2x+2x)+(8x^2+8x^2)+(9-4)=\\boxed{{16x^2+4x+5}}$.</answer><|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    
    args = parser.parse_args()
    
    data_source = 'xDAN2099/lighteval-MATH'
    dataset = load_dataset(data_source, trust_remote_code=True)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop('problem')
            formatted_question = make_prefix(question, template_type=args.template_type)
            
            if len(formatted_question.split()) > 300:
                print(f"Skipping example {idx} because it exceeds 300 words")
                formatted_question = ''  # filted out later
            
            answer = example.pop('solution')
            solution = extract_solution(answer)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": formatted_question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=[])
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=[])
    
    # Remove examples with empty prompts
    train_dataset = train_dataset.filter(lambda x: len(x['prompt'][0]['content']) > 0)
    test_dataset = test_dataset.filter(lambda x: len(x['prompt'][0]['content']) > 0)
    
    # Print out one example from the training set
    print(json.dumps(train_dataset[0], indent=2))
    
    # Print the number of examples in the training and testing sets
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of testing examples: {len(test_dataset)}")
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
