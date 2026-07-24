from typing import List, Dict, Any, Union
import re

def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein distance between two strings using dynamic programming."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def normalize_text(text: str) -> str:
    """Lowercases text and removes punctuation/excess whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())

def compute_exact_match(prediction: str, reference: str) -> float:
    """Computes Exact Match (EM) score (1.0 if identical, else 0.0)."""
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0

def compute_f1_score(prediction: str, reference: str) -> float:
    """Computes word-level F1 score overlap."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
        
    common = set(pred_tokens) & set(ref_tokens)
    num_same = sum(min(pred_tokens.count(w), ref_tokens.count(w)) for w in common)
    
    if num_same == 0:
        return 0.0
        
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_cer(prediction: str, reference: str) -> float:
    """Computes Character Error Rate (CER)."""
    ref_len = len(reference)
    if ref_len == 0:
        return 1.0 if len(prediction) > 0 else 0.0
    return levenshtein_distance(prediction, reference) / ref_len

def compute_wer(prediction: str, reference: str) -> float:
    """Computes Word Error Rate (WER)."""
    pred_words = prediction.split()
    ref_words = reference.split()
    
    ref_len = len(ref_words)
    if ref_len == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
        
    # We map lists of strings to lists of dummy characters to use the same Levenshtein function
    # or implement token-level edit distance directly
    word_to_char = {}
    def words_to_symbols(words):
        symbols = []
        for w in words:
            if w not in word_to_char:
                word_to_char[w] = chr(len(word_to_char) + 32)
            symbols.append(word_to_char[w])
        return "".join(symbols)

    pred_sym = words_to_symbols(pred_words)
    ref_sym = words_to_symbols(ref_words)
    
    return levenshtein_distance(pred_sym, ref_sym) / ref_len

def compute_anls(prediction: str, reference: str, threshold: float = 0.5) -> float:
    """Computes Average Normalized Levenshtein Similarity (ANLS) standard for DocVQA."""
    pred_norm = prediction.lower().strip()
    ref_norm = reference.lower().strip()
    
    max_len = max(len(pred_norm), len(ref_norm))
    if max_len == 0:
        return 1.0
        
    dist = levenshtein_distance(pred_norm, ref_norm)
    similarity = 1.0 - (dist / max_len)
    
    return similarity if similarity >= threshold else 0.0

def compute_batch_metrics(predictions: List[str], references: List[str], task: str = "vqa") -> Dict[str, float]:
    """Computes and aggregates metrics for a batch of predictions and references."""
    total_em = 0.0
    total_f1 = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_anls = 0.0
    
    count = len(predictions)
    if count == 0:
        return {}
        
    for pred, ref in zip(predictions, references):
        total_em += compute_exact_match(pred, ref)
        total_f1 += compute_f1_score(pred, ref)
        total_cer += compute_cer(pred, ref)
        total_wer += compute_wer(pred, ref)
        total_anls += compute_anls(pred, ref)
        
    metrics = {
        "exact_match": total_em / count,
        "f1": total_f1 / count,
        "cer": total_cer / count,
        "wer": total_wer / count
    }
    
    if task == "vqa":
        metrics["anls"] = total_anls / count
        
    return metrics
