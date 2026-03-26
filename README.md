# AI_Based_Cyber_Threat_Detection_System_for_Enterprises
# HOW TO RUN THE EXPERIMENT — Step-by-Step Guide
## AI-Based Cyber Threat Detection System for Enterprise Networks

---

## STEP 1 — Install Python (if not installed)
Download from: https://www.python.org/downloads/
Recommended version: Python 3.9 or 3.10
✔ During install, tick "Add Python to PATH"

---

## STEP 2 — Install required libraries
Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:

```
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn imbalanced-learn openpyxl
```

This takes 3–5 minutes. Do it once only.

---

## STEP 3 — Download the datasets (FREE)

### Dataset 1: CICIDS2017
1. Go to: https://www.unb.ca/cic/datasets/ids-2017.html
2. Scroll to "Download" and click "MachineLearningCSV.zip"
3. Extract the ZIP
4. You will get several CSV files like:
   - Monday-WorkingHours.pcap_ISCX.csv
   - Tuesday-WorkingHours.pcap_ISCX.csv
   - Wednesday-workingHours.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   - (and more)

### Dataset 2: UNSW-NB15
1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download these two files:
   - UNSW_NB15_training-set.csv
   - UNSW_NB15_testing-set.csv

---

## STEP 4 — Set up your folder structure

Create a folder anywhere on your PC, for example on Desktop: "cyber_research"

Inside it, create this exact structure:

```
cyber_research/
├── cyber_threat_detection_experiment.py    ← the Python script
├── data/
│   ├── CICIDS2017/
│   │   ├── Monday-WorkingHours.pcap_ISCX.csv
│   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │   └── (all other CICIDS CSV files here)
│   └── UNSW_NB15/
│       ├── UNSW_NB15_training-set.csv
│       └── UNSW_NB15_testing-set.csv
```

---

## STEP 5 — Run the script

Open Command Prompt / Terminal and type:

```
cd Desktop/cyber_research
python cyber_threat_detection_experiment.py
```

Press Enter and wait. The script will print progress as it runs.

**Expected run time on a standard PC:**
- CICIDS2017 preprocessing: ~2–4 minutes
- UNSW-NB15 preprocessing:  ~1–2 minutes
- Random Forest training:    ~3–5 minutes
- SVM training:              ~5–10 minutes
- LSTM training:             ~10–20 minutes
- CNN-LSTM training:         ~15–25 minutes
- Total:                     ~40–60 minutes

---

## STEP 6 — Collect your results

After the script finishes, a new folder called "results/" will appear:

```
results/
├── METRICS_LOG_FOR_PAPER.txt               ← Open this FIRST
├── Table3_Model_Performance_Comparison.csv
├── Table4_PerClass_Detection_CICIDS2017.csv
├── Table4_PerClass_Detection_UNSW-NB15.csv
├── Table5_Computational_Cost.csv
├── Figure1_ROC_Curves.png
├── Figure2_Confusion_Matrices.png
├── Figure3_Training_Curves.png
└── Figure4_F1_Comparison.png
```

Open METRICS_LOG_FOR_PAPER.txt — it lists every number
(accuracy, F1, AUC, etc.) ready to copy into your paper.

---

## STEP 7 — Fill your paper prompts

Take the numbers from METRICS_LOG_FOR_PAPER.txt and paste them
into the placeholders in your research paper prompts:

- [INSERT YOUR ACTUAL RESULTS HERE]   → use Table 3 values
- [INSERT FINDINGS SUMMARY]           → top 3 lines from the log
- [INSERT OBJECTIVES]                 → from your Introduction section

Insert the PNG figures directly into your Word/LaTeX document
as Figure 1, 2, 3, 4.

---

## TROUBLESHOOTING

| Problem | Solution |
|---|---|
| "No CSV files found" | Check folder path in Step 4 exactly |
| "ModuleNotFoundError" | Re-run the pip install command in Step 2 |
| Script very slow | Reduce MAX_SAMPLES in the script from 100_000 to 50_000 |
| Out of memory error | Reduce MAX_SAMPLES and BATCH_SIZE in the script |
| TensorFlow GPU warning | Safe to ignore — runs fine on CPU |

---

## NEED HELP?
If you get an error message, copy the full error text and share it.
The issue can be fixed quickly.
