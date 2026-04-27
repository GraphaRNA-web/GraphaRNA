import sys
import os
import math
from grapharna.preprocess_rna_pdb import get_dotseq_from_pdb

def extract_pairs(pdb_path):
    if not os.path.exists(pdb_path):
        return set()
        
    result = get_dotseq_from_pdb(pdb_path)
    if not result:
        return set()

    dot = ""
    if isinstance(result, tuple) and len(result) >= 2:
        dot = result[1]
    elif isinstance(result, str):
        # Rozbijamy po enterach i bierzemy TYLKO ostatnią linijkę (struktura)
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        dot = lines[-1]
        
    # Uniwersalny parser stosowy
    pairs = set()
    opening = "([{<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    closing = ")]}>abcdefghijklmnopqrstuvwxyz"
    match_map = {c: o for o, c in zip(opening, closing)}
    
    stacks = {o: [] for o in opening}
    
    valid_idx = 0
    for char in dot:
        if char in [' ', '&', '+', '-']:
            continue
            
        if char in opening:
            stacks[char].append(valid_idx)
        elif char in closing:
            o_char = match_map[char]
            if stacks[o_char]:
                start = stacks[o_char].pop()
                pairs.add((start, valid_idx))
        valid_idx += 1
        
    return pairs

def CalculateF1Inf(target: set, model: set):
    tp = len(target & model)
    fp = len(model - target)
    fn = len(target - model)
    
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 1
    
    inf = math.sqrt(precision * recall)
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    return {"inf": inf, "f1": f1}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
        
    ref_path = sys.argv[1]
    pred_path = sys.argv[2]
    
    ref_pairs = extract_pairs(ref_path)
    pred_pairs = extract_pairs(pred_path)
    
    metrics = CalculateF1Inf(ref_pairs, pred_pairs)
    
    print(f"INF: {metrics['inf']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")