# Audmind Dataset Creation Scripts

This repository contains all the scripts and notebooks used to create the **Audmind** dataset for mental health analysis from YouTube audio sources. The dataset creation pipeline involves audio extraction, transcription, conversation generation, task classification, QAR (Question-Answer-Reasoning) generation, and final audio datapoint formatting.

---

## 📁 Folder Structure

Each script contributes to a specific step of the dataset creation process:

### 1. **Audio Extraction**
- `download_audios.py`  
  Extracts raw audio chunks from YouTube links listed in `URL_sources.txt`.  
  ➤ Output: `audiochunks_all/`

---

### 2. **Transcription**
- `audiochunk2transcription_MentalHealth.py`  
  Converts audio chunks to transcriptions using **Whisper ASR** with diarization.  
  ➤ Output: `Transcriptions_all/`

- `audio2transcription_MentalHealth.ipynb`  
  Jupyter version for quick testing/debugging of the transcription step.

---

### 3. **Conversation Generation**
- `transcriptions2conversations.py`  
  Merges transcript chunks into long-form conversation-style text.  
  ➤ Output: `Transcriptions_conversations/`

---

### 4. **Task Classification (Inspired by MentaLLaMA)**
- `transcript_conversations2classification.py`  
  Classifies conversations into Task1, Task2, Task3, and Task4.  
  ➤ Output: `Transcriptions_classified_all/`

---

### 5. **QAR Generation**
- `transcript_classifications2qar.py`  
  Generates Question-Answer-Reasoning (QAR) tuples from classified conversations.  
  ➤ Output: `Transcriptions_qar/`

---

### 6. **Final Audio Datapoint Extraction**
- `audiodataset_extraction.py`  
  Crops audio into segments of interest and names them in the format:  
  `<audio_id>_chunk<#>_data<#>_task<#>.wav`  
  ➤ Output: `final_audio_dataset/`

- `audio_datapoints_extraction.ipynb`  
  Notebook version for testing/debugging.

---

### 7. **Metadata CSV Creation**
- `make_csv_for_audio_datapoints.py`  
  Generates `structured_output.csv` with `[file_id, Question, Answer, Reasoning]`.

- `make_final_complete_csv_with_text.py`  
  Generates `structured_output_with_text.csv` with `[file_id, text, Question, Answer, Reasoning]`.

---

### 8. **Train/Test Split and Preprocessing**
- `make_test_dataset.ipynb`  
  Splits the `final_audio_dataset` into `train_mentalhealth/` and `test_mentalhealth/`.

- `convert_test_data_to_16khz.py`  
  Converts test audio files to 16 kHz sampling rate.  
  ➤ Output: `test_mentalhealth_16kHz/`

---

## 📦 Outputs Summary

| Folder/File                        | Description |
|----------------------------------|-------------|
| `audiochunks_all/`               | Raw extracted audio parts per ID |
| `Transcriptions_all/`           | Part-wise transcriptions |
| `Transcriptions_conversations/` | Complete conversation-style transcripts |
| `Transcriptions_classified_all/`| Task-classified conversations |
| `Transcriptions_qar/`           | Generated QAR data |
| `final_audio_dataset/`          | Final audio datapoints for model input |
| `structured_output.csv`         | QAR metadata |
| `structured_output_with_text.csv` | Text + QAR metadata |
| `train_mentalhealth/`, `test_mentalhealth/`, `test_mentalhealth_16kHz/` | Final dataset splits |

---

## 🔗 Reference

Classification schema and methodology inspired by:  
**MentaLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models**

