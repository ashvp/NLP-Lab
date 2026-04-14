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