# Audmind Structured Data

This directory contains structured CSV files representing the processed Audmind dataset. These files are used for training, validation, testing, and evaluation across various models. The dataset is organized with and without raw transcription text depending on the use case.

---

## 📄 CSV File Overview

### 📌 Without Transcription Text
These files contain:  
**Columns:** `file_id`, `Question`, `Answer`, `Reasoning`

- `structured_output.csv` – Full dataset from `final_audio_dataset/`
- `train_mentalhealth.csv` – Training subset
- `test_mentalhealth.csv` – Testing subset

---

### 📌 With Transcription Text
These files contain:  
**Columns:** `file_id`, `text`, `Question`, `Answer`, `Reasoning`

- `audmind_full.csv` – Full dataset from `final_audio_dataset/`
- `train_audmind.csv` – Full training split
- `test_audmind.csv` – Full test split
- `trainsub_audmind.csv` – 90% of training data (used for training)
- `val_audmind.csv` – 10% of training data (used for validation)

---

## 🧠 Usage
These CSVs are used for:
- Baseline model inference preparation
- Training/validation split management
- Metric evaluation with ground truth
- Model performance comparison

