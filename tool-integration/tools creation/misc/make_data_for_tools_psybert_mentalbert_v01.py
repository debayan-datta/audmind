import numpy as np
import pandas as pd
import librosa
import subprocess
import tempfile
import soundfile as sf
from scipy.signal import find_peaks
from scipy import fft, signal

from huggingface_hub import login
login(token="HF-TOKEN")

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import partial

# 1. Load every model and tokenizer once
psychbert_model_name = "./psychbert-finetuned-multiclass"
psychbert_tokenizer = AutoTokenizer.from_pretrained(psychbert_model_name)
psychbert_model     = AutoModelForSequenceClassification.from_pretrained(psychbert_model_name)
psychbert_model.eval()

mentalbert_model_name = "mental/mental-bert-base-uncased"
mentalbert_tokenizer = AutoTokenizer.from_pretrained(mentalbert_model_name)
mentalbert_model     = AutoModelForSequenceClassification.from_pretrained(mentalbert_model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
mentalbert_model.to(device).eval()

v01_name = "tahaenesaslanturk/mental-health-classification-v0.1"
v01_tokenizer = AutoTokenizer.from_pretrained(v01_name)
v01_model     = AutoModelForSequenceClassification.from_pretrained(v01_name)
v01_model.eval()

# 2. Refactor functions to reuse loaded instances
def classification_psychbert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1).item()
    return {
        0: "Negative / unrelated to mental health",
        1: "Mental illnesses",
        2: "Anxiety",
        3: "Depression",
        4: "Social anxiety",
        5: "Loneliness"
    }[pred]

def extract_mentalbert_outputs(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = F.softmax(logits, dim=-1)
    pred       = torch.argmax(probs, dim=-1).item()
    confidence = torch.max(probs).item()
    return pd.Series({
        'mentalbert_depression': pred,
        'mentalbert_conf': confidence
    })

def classify_disorder_v01(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return model.config.id2label[pred]


# 3. Main application
if __name__ == "__main__":
    df = pd.read_csv("/data/amey_2311cs10/debayan/test_audmind.csv")
    df['file_id'] = "/data/amey_2311cs10/debayan/test_mentalhealth_16kHz/" + df['file_id'] + ".wav"
    
    # Bind loaded models into callables
    psych_fn = partial(classification_psychbert, model=psychbert_model, tokenizer=psychbert_tokenizer)
    mental_fn = partial(extract_mentalbert_outputs,
                         model=mentalbert_model,
                         tokenizer=mentalbert_tokenizer,
                         device=device)
    v01_fn = partial(classify_disorder_v01, model=v01_model, tokenizer=v01_tokenizer)
    
    # Apply across all rows without reloading
    df['pyschbert_classification'] = df['text'].apply(psych_fn)
    df[['mentalbert_depression','mentalbert_conf']] = df['text'].apply(mental_fn)
    df['v01_classification'] = df['text'].apply(v01_fn)
    
    # Optionally save results
    df.to_csv("test_data_for_tools_with_pysbert_mentalbert_v01.csv", index=False)
