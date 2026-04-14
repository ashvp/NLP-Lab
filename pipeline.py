
import subprocess
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load SPECTER model
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()

def run_c_alignment(text1, text2):
    # Call smith_waterby.exe - C
    process = subprocess.Popen(['./smith_waterby.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, _ = process.communicate(input=f"{text1}\n{text2}\n")
    
    match = re.search(r"Local Alignment Score: (\d+)", stdout)
    if match:
        raw_score = int(match.group(1))
        # Normalize: divide by 2 * min_length (max possible match score if match=2)
        norm_score = raw_score / (2 * min(len(text1), len(text2)))
        return min(norm_score, 1.0)
    return 0.0

def run_c_citation(text):
    # Call citation.exe
    process = subprocess.Popen(['./citation.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, _ = process.communicate(input=f"{text}\n")
    match = re.search(r"Citations detected: (\d+)", stdout)
    return int(match.group(1)) if match else 0

def calculate_pipeline(doc1, doc2):
    # Weights - random
    ALPHA, BETA, GAMMA, DELTA, EPSILON = 0.3, 0.2, 0.2, 0.2, 0.1

    # 1. S_sem: Global Semantic Similarity
    emb1 = get_embedding(doc1)
    emb2 = get_embedding(doc2)
    s_sem = cosine_similarity([emb1], [emb2])[0][0]

    # 2. S_struct: Structural Similarity (Paragraph alignment)
    paras1 = [p.strip() for p in doc1.split('\n') if len(p.strip()) > 20]
    paras2 = [p.strip() for p in doc2.split('\n') if len(p.strip()) > 20]
    
    if paras1 and paras2:
        embs1 = np.array([get_embedding(p) for p in paras1])
        embs2 = np.array([get_embedding(p) for p in paras2])
        sim_matrix = cosine_similarity(embs1, embs2)
        # Average of maximum paragraph similarities for structural coherence
        s_struct = np.mean(np.max(sim_matrix, axis=1))
    else:
        s_struct = s_sem

    # 3. S_align: Local Alignment via C
    s_align = run_c_alignment(doc1[:1000], doc2[:1000]) # Limit to C's MAX_LEN - have to change

    # 4. S_cite: Citation Consistency
    cite_count = run_c_citation(doc1)
    # Risk is higher if similarity is high but citations are low
    # Here we simplify: penalty if cite_count is 0 despite high similarity
    s_cite = 1.0 if (s_sem > 0.7 and cite_count == 0) else (1.0 - min(cite_count/5.0, 1.0))

    # 5. S_common: Common Knowledge Penalty - should be huge corpus of all research docs
    common_phrases = [
        "The paper is organized as follows.",
        "Recent advances in deep learning have shown great promise.",
        "Future work will focus on improving the performance.",
        "The Smith-Waterman algorithm is used for sequence alignment.",
        "Artificial Intelligence is a rapidly evolving field."
    ]
    common_embs = np.array([get_embedding(p) for p in common_phrases])
    doc_embs = np.array([get_embedding(p) for p in paras1]) if paras1 else [emb1]
    common_sims = cosine_similarity(doc_embs, common_embs)
    s_common = np.mean(np.max(common_sims, axis=1))

    # Final Risk Score R
    r = (ALPHA * s_sem) + (BETA * s_struct) + (GAMMA * s_align) + (DELTA * s_cite) - (EPSILON * s_common)
    
    return {
        "Risk Score (R)": max(0.0, min(r, 1.0)),
        "S_sem": s_sem,
        "S_struct": s_struct,
        "S_align": s_align,
        "S_cite": s_cite,
        "S_common": s_common
    }

if __name__ == "__main__":
    doc_a = """
    Artificial Intelligence has revolutionized the way we interact with technology.
    Deep learning models, especially transformers, have achieved state-of-the-art results
    in natural language processing. The Smith-Waterman algorithm remains a fundamental
    technique for local sequence alignment.
    """
    
    doc_b = """
    AI is changing how humans use tech. Neural networks like transformers 
    are leading the way in NLP tasks. We use Smith-Waterman for 
    local alignment of sequences in this study.
    """

    results = calculate_pipeline(doc_a, doc_b)
    print("\n--- Plagiarism Detection Report ---")
    for k, v in results.items():
        print(f"{k:20}: {v:.4f}")
