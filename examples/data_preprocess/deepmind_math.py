import os
import json
import argparse
from datasets import Dataset
from tqdm import tqdm


def read_dm_math_data(folder_path):
    """Read and parse .txt files in a given folder."""
    samples = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i in range(0, len(lines) - 1, 2):  # Question on even, Answer on next line
                    question = lines[i].strip()
                    answer = lines[i + 1].strip()
                    samples.append((question, answer))
    return samples


def make_prefix(question, template_type):
    """Format the question with a prefix for different model templates."""
    if template_type == "base":
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. For example, <answer> 42 </answer> if the solution is a number, or <answer> 5/22 </answer> if the solution is a fraction, or <answer> x**2 + 3*y**2 + 5*z + 2 </answer> if the solution is an expression, or <answer> yes </answer> if the solution is a yes/no answer.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == "qwen-instruct":
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.<|im_end|>\n<|im_start|>user\n{question} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. For example, <answer> 42 </answer> if the solution is a number, or <answer> 5/22 </answer> if the solution is a fraction, or <answer> x**2 + 3*y**2 + 5*z + 2 </answer> if the solution is an expression, or <answer> yes </answer> if the solution is a yes/no answer.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


def process_data(source_folder, split, template_type):
    """Process math data and format it into JSON structure."""
    data = []
    samples = read_dm_math_data(source_folder)
    for idx, (question, answer) in enumerate(tqdm(samples, desc=f"Processing {split}")):
        formatted_question = make_prefix(question, template_type)
        solution = {
            "target": answer
        }
        data.append({
            "data_source": "dm_math",
            "prompt": [{
                "role": "user",
                "content": formatted_question,
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                "split": split,
                "index": idx,
            }
        })
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dm_math", help="Root directory for dm_math data")
    parser.add_argument("--output_dir", default="./dm_math/processed", help="Output directory for processed data")
    parser.add_argument("--template_type", type=str, default="base", choices=["base", "qwen-instruct"], help="Template type")
    parser.add_argument("--train_size", type=int, default=5000, help="Max number of training samples")
    parser.add_argument("--test_size", type=int, default=700, help="Max number of testing samples")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process training sets from all difficulty levels
    train_data = []
    for difficulty in ["train-easy", "train-medium", "train-hard"]:
        folder_path = os.path.join(args.data_dir, difficulty)
        train_data.extend(process_data(folder_path, split="train", template_type=args.template_type))

    # Split into train and test
    assert len(train_data) > args.train_size, f"Not enough data for train split, only have {len(train_data)}"
    train_set = train_data[: args.train_size]
    
    # Preview one example of training data
    print(json.dumps(train_set[0], indent=2))
    
    # Process test sets
    test_data = []
    for difficulty in ["extrapolate", "interpolate"]:
        folder_path = os.path.join(args.data_dir, difficulty)
        test_data.extend(process_data(folder_path, split="test", template_type=args.template_type))
        
    # Preview one example of testing data
    print(json.dumps(test_data[0], indent=2))

    # Split into train and test
    assert len(test_data) > args.test_size, f"Not enough data for test split, only have {len(test_data)}"
    test_set = test_data[: args.test_size]

    # Convert to Dataset and save as Parquet
    train_dataset = Dataset.from_list(train_set)
    test_dataset = Dataset.from_list(test_set)

    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.output_dir, "test.parquet"))

    print(f"Processed datasets saved in {args.output_dir}")
