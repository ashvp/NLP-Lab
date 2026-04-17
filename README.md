# Academic Plagiarism Detection — NLP Lab

A multi-signal plagiarism detection pipeline for academic papers combining SPECTER embeddings, n-gram alignment, and citation analysis.

---

## 🖥️ Streamlit Web App

A browser-based frontend for the pipeline is available in `app.py`.

### Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Compile the C executables** (requires GCC / MinGW on Windows)
```bash
gcc -O2 -o preprocessor.exe preprocessor.c
gcc -O2 -o citation.exe citation.c
gcc -O2 -o smith_waterby.exe smith_waterby.c
```

**3. Run the app**
```bash
streamlit run app.py
```

### Features
- Upload two PDFs **or** paste raw text directly
- Live risk score gauge with LOW / MEDIUM / HIGH verdict
- Radar chart and bar chart of all five pipeline components
- Full score breakdown: S_sem, S_struct, S_align, S_cite, S_common
- Shows final weight redistribution when no citations are detected
- Expandable preprocessed text preview and full formula calculation

### Pipeline Components

| Component | Method | Weight |
|-----------|--------|--------|
| S_sem — Semantic Similarity | SPECTER [CLS] cosine similarity | α = 0.45 |
| S_struct — Structural Consistency | Paragraph-level embedding alignment | β = 0.10 |
| S_align — N-gram Alignment | 6-gram Jaccard overlap | γ = 0.25 |
| S_cite — Citation Score | C-based citation counter | δ = 0.15 |
| S_common — Common Knowledge Penalty | Boilerplate phrase similarity | ε = 0.05 |

**Risk Score Formula:**
```
R = α·S_sem + β·S_struct + γ·S_align + δ·S_cite − ε·S_common
```

| Score Range | Verdict |
|-------------|---------|
| ≥ 0.65 | 🔴 HIGH RISK — likely plagiarism |
| 0.45 – 0.65 | 🟡 MEDIUM RISK — review recommended |
| < 0.45 | 🟢 LOW RISK — likely original |

---

## Experiment Results

# attention is all you need vs canny edge (only abstracts)
```
(.venv) PS D:\SNU\Semester 6\NLP\lab> python .\pipeline_max.py
Loading SPECTER model...
Loading weights: 100%|█████████████████████████████████████████████████████████| 199/199 [00:00<?, ?it/s]
BertModel LOAD REPORT from: allenai/specter
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
No command line arguments detected. Running in Demo Mode (Raw Text)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Running Plagiarism Detection Pipeline...
Calculating S_sem (Semantic Similarity)...
Calculating S_struct (Structural Consistency)...
Calculating S_align (Local Alignment via C)...
Calculating S_cite (Citation Analysis via C)...
No citations detected. Redistributing weights...
Calculating S_common (Common Knowledge Penalty)...

========================================
      PLAGIARISM DETECTION REPORT
========================================
Risk Score (R)      : 0.4173
S_sem               : 0.7303
S_struct            : 0.8219
S_align             : 0.0270
S_cite              : 0.0000
S_common            : 0.8266

Final Weights:
  ALPHA: 0.39
  BETA: 0.26
  GAMMA: 0.26
  DELTA: 0.00
========================================
```

# attention is all you need vs attention is all you need *rewrite* (abstracts)
```
(.venv) PS D:\SNU\Semester 6\NLP\lab> python .\pipeline_max.py
Loading SPECTER model...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████| 199/199 [00:00<00:00, 8863.40it/s]
BertModel LOAD REPORT from: allenai/specter
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
No command line arguments detected. Running in Demo Mode (Raw Text)...

Running Plagiarism Detection Pipeline...
Calculating S_sem (Semantic Similarity)...
Calculating S_struct (Structural Consistency)...
Calculating S_align (Local Alignment via C)...
Calculating S_cite (Citation Analysis via C)...
No citations detected. Redistributing weights...
Calculating S_common (Common Knowledge Penalty)...

========================================
      PLAGIARISM DETECTION REPORT
========================================
Risk Score (R)      : 0.6269
S_sem               : 0.9745
S_struct            : 0.8915
S_align             : 0.3745
S_cite              : 0.0000
S_common            : 0.7453

Final Weights:
  ALPHA: 0.39
  BETA: 0.26
  GAMMA: 0.26
  DELTA: 0.00
========================================
```

# attention is all you need vs nerf (PDFs)
```
(.venv) PS D:\SNU\Semester 6\NLP\lab> python .\pipeline_max.py ".\attention is all you need.pdf" ".\nerf.pdf"
Loading SPECTER model...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████| 199/199 [00:00<00:00, 12476.70it/s]
BertModel LOAD REPORT from: allenai/specter
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Processing PDF 1: .\attention is all you need.pdf...
Processing PDF 2: .\nerf.pdf...

Running Plagiarism Detection Pipeline...
Calculating S_sem (Semantic Similarity)...
Calculating S_struct (Structural Consistency)...
Calculating S_align (N-gram Alignment)...
Calculating S_cite (Citation Analysis via C)...
Calculating S_common (Common Knowledge Penalty)...

==========================================
       PLAGIARISM DETECTION REPORT
==========================================
Risk Score (R)        : 0.3406
S_sem                 : 0.6612
S_struct              : 0.8494
S_align               : 0.0000
S_cite                : 0.0000
S_common              : 0.8385
Citations Found       : 109

Final Weights:
  ALPHA: 0.450
  BETA: 0.100
  GAMMA: 0.250
  DELTA: 0.150
  EPSILON: 0.050
==========================================

Verdict: LOW RISK   — likely original
==========================================
```

# attention is all you need vs attention is all you need full *rewrite* (PDFs)
```
(.venv) PS D:\SNU\Semester 6\NLP\lab> python .\pipeline_max.py ".\attention is all you need.pdf" ".\ms.pdf"
Loading SPECTER model...
Loading weights: 100%|███████████████████████████████████████████████| 199/199 [00:00<00:00, 8750.32it/s]
BertModel LOAD REPORT from: allenai/specter
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Processing PDF 1: .\attention is all you need.pdf...
Processing PDF 2: .\ms.pdf...

Running Plagiarism Detection Pipeline...
Calculating S_sem (Semantic Similarity)...
Calculating S_struct (Structural Consistency)...
Calculating S_align (N-gram Alignment)...
Calculating S_cite (Citation Analysis via C)...
Calculating S_common (Common Knowledge Penalty)...

==========================================
       PLAGIARISM DETECTION REPORT
==========================================
Risk Score (R)        : 0.7579
S_sem                 : 1.0000
S_struct              : 0.9979
S_align               : 1.0000
S_cite                : 0.0000
S_common              : 0.8385
Citations Found       : 109

Final Weights:
  ALPHA: 0.450
  BETA: 0.100
  GAMMA: 0.250
  DELTA: 0.150
  EPSILON: 0.050
==========================================

Verdict: HIGH RISK  — likely plagiarism
=========================================
```

# Using BERT for embeddings 
## Attention is all you need vs attention is all you need full *rewrite* (PDFs)
```
(.venv) PS D:\SNU\Semester 6\NLP\lab> python .\end_sem.py ".\attention is all you need.pdf" ".\ms.pdf"    
Loading BERT model (bert-base-uncased)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████| 199/199 [00:00<00:00, 6050.72it/s]
BertModel LOAD REPORT from: bert-base-uncased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.seq_relationship.weight                | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Calculating S_sem (BERT Semantic Similarity)...
Calculating S_struct (Structural Consistency)...
Calculating S_align (Local Alignment via C)...
Calculating S_cite (Citation Analysis via C)...
Calculating S_common (Common Knowledge Penalty)...

========================================
      PLAGIARISM REPORT (BERT VERSION)
========================================
Risk Score (R)      : 0.4171
S_sem               : 1.0000
S_struct            : 0.9951
S_align             : 0.0011
S_cite              : 0.0000
S_common            : 0.8210

Final Weights:
  ALPHA: 0.30
  BETA: 0.20
  GAMMA: 0.20
  DELTA: 0.20
========================================
(.venv) PS D:\SNU\Semester 6\NLP\lab> 
```