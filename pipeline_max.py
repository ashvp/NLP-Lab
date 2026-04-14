import subprocess
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import fitz  # PyMuPDF
import os
import sys

# Load SPECTER model
print("Loading SPECTER model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

def get_embedding(text):
    if not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()

def preprocess_with_c(text_with_pages):
    if "---PAGE_BREAK---" not in text_with_pages:
        text_with_pages += "\n---PAGE_BREAK---\n"
    process = subprocess.Popen(
        ['./preprocessor.exe'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding='utf-8'
    )
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
    cleaned_text = preprocess_with_c(text_with_pages)
    return cleaned_text

# ── NEW: word-level n-gram overlap replaces character-level Smith-Waterman ──
def compute_ngram_alignment(text1, text2, n=6):
    """
    Sliding-window n-gram Jaccard overlap.
    Samples beginning / middle / end of each document and returns
    the maximum overlap found across all 3×3 window pairs.
    A paraphrase that preserves 6-word phrases will score noticeably
    higher than two completely unrelated documents.
    """
    def ngrams(text, n):
        words = re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
        return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))

    def get_windows(text, chars=3000):
        mid = len(text) // 2
        return [
            text[:chars],
            text[max(0, mid - chars // 2): mid + chars // 2],
            text[-chars:]
        ]

    max_score = 0.0
    for w1 in get_windows(text1):
        for w2 in get_windows(text2):
            ng1 = ngrams(w1, n)
            ng2 = ngrams(w2, n)
            if not ng1 or not ng2:
                continue
            overlap = len(ng1 & ng2) / min(len(ng1), len(ng2))
            max_score = max(max_score, overlap)
    return min(max_score, 1.0)

def run_c_citation(text):
    process = subprocess.Popen(
        ['./citation.exe'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding='utf-8'
    )
    stdout, _ = process.communicate(input=text)
    match = re.search(r"Citations detected: (\d+)", stdout)
    return int(match.group(1)) if match else 0

def calculate_pipeline(doc1, doc2):
    # Base weights  (refined after multi-paper calibration)
    ALPHA   = 0.45   # Semantic similarity  – most reliable signal
    BETA    = 0.10   # Structural consistency – noisy for abstracts, reduced
    GAMMA   = 0.25   # N-gram alignment      – catches paraphrase traces
    DELTA   = 0.15   # Citation score        – fires when citations present
    EPSILON = 0.05   # Common-knowledge penalty – halved to stop masking TPs

    print("Calculating S_sem (Semantic Similarity)...")
    emb1 = get_embedding(doc1)
    emb2 = get_embedding(doc2)
    s_sem = float(cosine_similarity([emb1], [emb2])[0][0])

    print("Calculating S_struct (Structural Consistency)...")
    paras1 = [p.strip() for p in doc1.split('\n') if len(p.strip()) > 20]
    paras2 = [p.strip() for p in doc2.split('\n') if len(p.strip()) > 20]

    if paras1 and paras2:
        paras1 = paras1[:50]
        paras2 = paras2[:50]
        embs1 = np.array([get_embedding(p) for p in paras1])
        embs2 = np.array([get_embedding(p) for p in paras2])
        sim_matrix = cosine_similarity(embs1, embs2)
        s_struct = float(np.mean(np.max(sim_matrix, axis=1)))
    else:
        s_struct = s_sem

    print("Calculating S_align (N-gram Alignment)...")
    s_align = compute_ngram_alignment(doc1, doc2)

    print("Calculating S_cite (Citation Analysis via C)...")
    cite_count = run_c_citation(doc1)

    if cite_count == 0:
        print("No citations detected. Redistributing DELTA weight...")
        # Redistribute proportionally to remaining weights
        total = ALPHA + BETA + GAMMA   # = 0.80
        ALPHA += DELTA * (ALPHA / total)
        BETA  += DELTA * (BETA  / total)
        GAMMA += DELTA * (GAMMA / total)
        DELTA  = 0.0
        s_cite = 0.0
    else:
        # Fewer citations relative to high similarity → higher risk
        s_cite = 1.0 - min(cite_count / 10.0, 1.0)

    print("Calculating S_common (Common Knowledge Penalty)...")
    common_phrases = [
        "The paper is organized as follows.",
        "Recent advances in deep learning have shown great promise.",
        "Future work will focus on improving the performance.",
        "The Smith-Waterman algorithm is used for sequence alignment.",
        "Artificial Intelligence is a rapidly evolving field."
    ]
    common_embs = np.array([get_embedding(p) for p in common_phrases])
    doc_embs = np.array([get_embedding(p) for p in paras1]) if paras1 else np.array([emb1])
    common_sims = cosine_similarity(doc_embs, common_embs)
    s_common = float(np.mean(np.max(common_sims, axis=1)))

    r = (ALPHA * s_sem) + (BETA * s_struct) + (GAMMA * s_align) + (DELTA * s_cite) - (EPSILON * s_common)

    return {
        "Risk Score (R)": max(0.0, min(r, 1.0)),
        "S_sem":    s_sem,
        "S_struct": s_struct,
        "S_align":  s_align,
        "S_cite":   s_cite,
        "S_common": s_common,
        "Citations Found": cite_count,
        "Final Weights": {
            "ALPHA":   f"{ALPHA:.3f}",
            "BETA":    f"{BETA:.3f}",
            "GAMMA":   f"{GAMMA:.3f}",
            "DELTA":   f"{DELTA:.3f}",
            "EPSILON": f"{EPSILON:.3f}"
        }
    }

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pdf1 = sys.argv[1]
        pdf2 = sys.argv[2]
        print(f"Processing PDF 1: {pdf1}...")
        doc_a = extract_pdf_text(pdf1)
        print(f"Processing PDF 2: {pdf2}...")
        doc_b = extract_pdf_text(pdf2)
    else:
        print("No command line arguments detected. Running in Demo Mode (Raw Text)...")

        # --- EDIT THESE STRINGS FOR DEMO ---
        doc_a = """
        The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English
to-German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8 after
training for 3.5 days on eight GPUs, a small fraction of the training costs of the
best models from the literature. We show that the Transformer generalizes well to
other tasks by applying it successfully to English constituency parsing both with
large and limited training data
        """

        doc_b = """
        Most sequence transduction models rely on complex recurrent or convolutional neural networks built with encoder-decoder structures. The top-performing ones further enhance this setup using attention mechanisms to connect the encoder and decoder.

This work introduces the Transformer, a much simpler architecture that relies entirely on attention, eliminating both recurrence and convolution altogether. Experiments on machine translation tasks show that this approach not only improves translation quality but also enables greater parallelization and significantly faster training.

On the WMT 2014 English-German task, the model achieves a BLEU score of 28.4, outperforming previous best results-including ensemble methods-by more than 2 BLEU points. For the English-French task, it sets a new single-model state-of-the-art BLEU score of 41.8, trained in just 3.5 days on eight GPUs, which is far more efficient than earlier approaches.

Additionally, the Transformer demonstrates strong generalization by performing well on other tasks, such as English constituency parsing, across both large and small datasets.
        """
        # ------------------------------------

        doc_a = preprocess_with_c(doc_a)
        doc_b = preprocess_with_c(doc_b)

    print("\nRunning Plagiarism Detection Pipeline...")
    results = calculate_pipeline(doc_a, doc_b)

    print("\n" + "="*42)
    print("       PLAGIARISM DETECTION REPORT")
    print("="*42)
    for k, v in results.items():
        if k == "Final Weights":
            print(f"\n{k}:")
            for wk, wv in v.items():
                print(f"  {wk}: {wv}")
        elif isinstance(v, int):
            print(f"{k:22}: {v}")
        else:
            print(f"{k:22}: {v:.4f}")
    print("="*42)

    # Verdict
    r = results["Risk Score (R)"]
    if r >= 0.65:
        verdict = "HIGH RISK  — likely plagiarism"
    elif r >= 0.45:
        verdict = "MEDIUM RISK — review recommended"
    else:
        verdict = "LOW RISK   — likely original"
    print(f"\nVerdict: {verdict}")
    print("="*42)