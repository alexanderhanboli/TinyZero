import numpy as np
from typing import Tuple
# from bert_score import score
from rouge_score import rouge_scorer
# from nltk.translate import meteor_score


def _compute_f1(predicted: str, gold: str) -> float:
    """Computes the F1-score between two strings."""
    predicted_tokens = set(predicted.split())
    gold_tokens = set(gold.split())
    intersection = len(predicted_tokens.intersection(gold_tokens))
    if not predicted_tokens:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_tokens))
    if not gold_tokens:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_tokens))
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def _compute_exact_match(predicted: str, gold: str) -> float:
    """Computes exact match score."""
    return float(predicted.strip().lower() == gold.strip().lower())

def _compute_rouge(predicted: str, gold: str) -> float:
    """Computes ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(gold, predicted)['rougeL'].fmeasure

# def _compute_meteor(predicted: str, gold: str) -> float:
#     """Computes METEOR score."""
#     return meteor_score.single_meteor_score(gold, predicted)

# def _compute_bertscore(predicted: str, gold: str) -> float:
#     """Computes BERTScore."""
#     P, R, F1 = score([predicted], [gold], num_layers=12, model_type="bert-base-uncased", lang="en")
#     return F1.item()

def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    Computes a reward score between 0 and 1 based on multiple similarity metrics.
    Higher scores indicate better alignment with the ground truth.
    """
    f1 = _compute_f1(solution_str, ground_truth)
    exact_match = _compute_exact_match(solution_str, ground_truth)
    rouge = _compute_rouge(solution_str, ground_truth)
    # meteor = _compute_meteor(solution_str, ground_truth)
    # bert_score = _compute_bertscore(solution_str, ground_truth)
    
    # Weighted sum of the scores
    reward = (0.33 * f1 + 0.34 * exact_match + 0.33 * rouge)
    return round(reward, 2)
