import os
import argparse
import json
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# 加载 Qwen 的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

def make_prefix(question, tables, paragraph, template_type):
    table_str = "These are the relevant tables to the paragraph:\n"
    for table in tables:
        table_str += table["latex"] + "\n"
        
    question = f"This is a paragraph from an academic report:\n{paragraph}\n{table_str}\n\nThe question is:\n{question}"

    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

def process_json_files(input_dir, output_dir, template_type):
    train_data = []
    test_data = []
    token_counts = []
    index = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)  # 假设 JSON 文件包含数据项列表
                for data in tqdm(data_list, desc=f"Processing {filename}", total=len(data_list)):
                    question = data["question"]
                    paragraph = data["paragraph"]["text"]
                    tables = data["tables"]
                    answer = data["answer"]
                    # 如果answer是list，join with“， ”
                    if isinstance(answer, list):
                        answer = ", ".join(answer)
                    # Check if answer is a string, if not, print out its value and warning, and convert it to string
                    if not isinstance(answer, str):
                        print(f"Warning: answer is not a string. Value: {answer}")
                        answer = str(answer)
                    formatted_question = make_prefix(question, tables, paragraph, template_type)
                    
                    # 计算 token 数
                    tokenized = tokenizer(formatted_question, return_tensors="pt")
                    num_tokens = tokenized.input_ids.shape[1]
                    token_counts.append(num_tokens)
                    
                    transformed = {
                        "type": data["question_type"],
                        "data_source": "scitat",
                        "prompt": [{
                            "role": "user",
                            "content": formatted_question
                        }],
                        "ability": "table-ds",
                        "reward_model": {
                            "ground_truth": answer,
                            "style": "rule"
                        },
                        "extra_info": {
                            "index": index,
                            "split": "test" if "test" in filename else "train"
                        },
                        "token_count": num_tokens
                    }
                    
                    if "test" in filename:
                        test_data.append(transformed)
                    else:
                        train_data.append(transformed)
                    index += 1
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_train = os.path.join(output_dir, "train.parquet")
    output_test = os.path.join(output_dir, "test.parquet")
    pd.DataFrame(train_data).to_parquet(output_train)
    pd.DataFrame(test_data).to_parquet(output_test)
    
    # 统计 token 数目分布
    plt.hist(token_counts, bins=30, edgecolor='black')
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Distribution of Token Counts in Formatted Questions")
    plt.show()
    
    # 最多的token数
    print(f"Maximum number of tokens: {max(token_counts)}")
    
    # 打印 JSON 数据示例
    print(json.dumps(train_data[0], indent=4))
    
    # 训练和测试集的样本数
    print(f"Number of examples in the training set: {len(train_data)}")
    print(f"Number of examples in the testing set: {len(test_data)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/home/hanboli/projects/SciTaT/dataset')
    parser.add_argument('--output_dir', default='/home/hanboli/efs/tinyzero/scitat')
    parser.add_argument('--template_type', type=str, default='base')
    
    args = parser.parse_args()
    
    process_json_files(args.input_dir, args.output_dir, args.template_type)
