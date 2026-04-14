import subprocess
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import fitz  # PyMuPDF
import os
import sys

# Load BERT model
print("Loading BERT model (bert-base-uncased)...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embedding(text):
    if not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    # Using [CLS] token embedding
    return output.last_hidden_state[:, 0, :].squeeze().numpy()

def preprocess_with_c(text_with_pages):
    if "---PAGE_BREAK---" not in text_with_pages:
        text_with_pages += "\n---PAGE_BREAK---\n"
    process = subprocess.Popen(['./preprocessor.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    stdout, stderr = process.communicate(input=text_with_pages)
    if stderr:
        print(f"Preprocessor Error: {stderr}")
    return stdout

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text_with_pages = ""
    for page in doc:
        text_with_pages += page.get_text() + "\n---PAGE_BREAK---\n"
    doc.close()
    return preprocess_with_c(text_with_pages)

def run_c_alignment(text1, text2):
    t1 = text1[:4990].replace('\n', ' ')
    t2 = text2[:4990].replace('\n', ' ')
    process = subprocess.Popen(['./smith_waterby.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    stdout, _ = process.communicate(input=f"{t1}\n{t2}\n")
    match = re.search(r"Local Alignment Score: (\d+)", stdout)
    if match:
        raw_score = int(match.group(1))
        norm_score = raw_score / (2 * min(len(t1), len(t2))) if min(len(t1), len(t2)) > 0 else 0
        return min(norm_score, 1.0)
    return 0.0

def run_c_citation(text):
    process = subprocess.Popen(['./citation.exe'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    stdout, _ = process.communicate(input=text)
    match = re.search(r"Citations detected: (\d+)", stdout)
    return int(match.group(1)) if match else 0

def calculate_pipeline(doc1, doc2):
    # Base Weights
    ALPHA, BETA, GAMMA, DELTA, EPSILON = 0.3, 0.2, 0.2, 0.2, 0.1

    print("Calculating S_sem (BERT Semantic Similarity)...")
    emb1 = get_embedding(doc1)
    emb2 = get_embedding(doc2)
    s_sem = cosine_similarity([emb1], [emb2])[0][0]

    print("Calculating S_struct (Structural Consistency)...")
    paras1 = [p.strip() for p in doc1.split('\n') if len(p.strip()) > 20]
    paras2 = [p.strip() for p in doc2.split('\n') if len(p.strip()) > 20]
    
    if paras1 and paras2:
        paras1 = paras1[:50]
        paras2 = paras2[:50]
        embs1 = np.array([get_embedding(p) for p in paras1])
        embs2 = np.array([get_embedding(p) for p in paras2])
        sim_matrix = cosine_similarity(embs1, embs2)
        s_struct = np.mean(np.max(sim_matrix, axis=1))
    else:
        s_struct = s_sem

    print("Calculating S_align (Local Alignment via C)...")
    s_align = run_c_alignment(doc1, doc2)

    print("Calculating S_cite (Citation Analysis via C)...")
    cite_count = run_c_citation(doc1)
    
    if cite_count == 0:
        print("No citations detected. Redistributing weights...")
        ALPHA += DELTA * (0.3 / 0.7)
        BETA += DELTA * (0.2 / 0.7)
        GAMMA += DELTA * (0.2 / 0.7)
        DELTA = 0.0
        s_cite = 0.0 
    else:
        s_cite = 1.0 - min(cite_count/10.0, 1.0)

    print("Calculating S_common (Common Knowledge Penalty)...")
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

    r = (ALPHA * s_sem) + (BETA * s_struct) + (GAMMA * s_align) + (DELTA * s_cite) - (EPSILON * s_common)
    
    return {
        "Risk Score (R)": max(0.0, min(r, 1.0)),
        "S_sem": s_sem,
        "S_struct": s_struct,
        "S_align": s_align,
        "S_cite": s_cite,
        "S_common": s_common,
        "Final Weights": {"ALPHA": f"{ALPHA:.2f}", "BETA": f"{BETA:.2f}", "GAMMA": f"{GAMMA:.2f}", "DELTA": f"{DELTA:.2f}"}
    }

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        doc_a = extract_pdf_text(sys.argv[1])
        doc_b = extract_pdf_text(sys.argv[2])
    else:
        print("Running in BERT Demo Mode (Raw Text)...")
        doc_a = """
        The transformer architecture has fundamentally changed natural language processing. 
        By using self-attention mechanisms, models like BERT and GPT can capture long-range 
        dependencies in text more effectively than previous RNN-based approaches. 
        """
        doc_b = """
        Language processing was transformed by the introduction of transformer models. 
        These architectures utilize self-attention to manage distant word relationships 
        better than recurrent neural networks did. 
        """
        doc_a = preprocess_with_c(doc_a)
        doc_b = preprocess_with_c(doc_b)

    results = calculate_pipeline(doc_a, doc_b)
    
    print("\n" + "="*40)
    print("      PLAGIARISM REPORT (BERT VERSION)")
    print("="*40)
    for k, v in results.items():
        if k == "Final Weights":
            print(f"\n{k}:")
            for wk, wv in v.items():
                print(f"  {wk}: {wv}")
        else:
            print(f"{k:20}: {v:.4f}")
    print("="*40)
