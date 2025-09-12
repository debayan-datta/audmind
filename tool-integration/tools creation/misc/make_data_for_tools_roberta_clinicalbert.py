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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. ClinicalBERT Manager
class ClinicalBERTManager:
    _instance = None
    
    def __new__(cls, device="auto"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance
    
    def initialize(self):
        if self.model is None:
            try:
                model_name = "emilyalsentzer/Bio_ClinicalBERT"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"ClinicalBERT init error: {e}")
                raise

    def classify(self, text: str, max_length=512) -> dict:
        if self.model is None:
            self.initialize()
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predictions_np = predictions.cpu().numpy()[0]
        return {
            "predicted_class": int(np.argmax(predictions_np)),
            "confidence": float(predictions_np[np.argmax(predictions_np)])
        }

# 2. RoBERTa Sentiment Manager
class RobertaSentimentManager:
    _instance = None
    
    def __new__(cls, device="auto"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance
    
    def initialize(self):
        if self.model is None:
            try:
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"RoBERTa init error: {e}")
                raise

    def analyze(self, text: str, max_length=512) -> dict:
        if self.model is None:
            self.initialize()
            
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        predictions_np = predictions.cpu().numpy()[0]
        sentiment_labels = ["negative", "neutral", "positive"]
        return {
            "predicted_sentiment": sentiment_labels[np.argmax(predictions_np)],
            "confidence": float(predictions_np[np.argmax(predictions_np)])
        }

# 3. DataFrame Processing Functions
def extract_clinicalbert_outputs(text: str) -> pd.Series:
    manager = ClinicalBERTManager()
    results = manager.classify(text)
    return pd.Series({
        'clinicalbert_class': results['predicted_class'],
        'clinicalbert_conf': results['confidence']
    })

def extract_roberta_outputs(text: str) -> pd.Series:
    manager = RobertaSentimentManager()
    results = manager.analyze(text)
    return pd.Series({
        'roberta_sentiment': results['predicted_sentiment'],
        'roberta_conf': results['confidence']
    })



if __name__ == "__main__":
    df = pd.read_csv("/data/amey_2311cs10/debayan/test_audmind.csv")
    df['file_id'] = "/data/amey_2311cs10/debayan/test_mentalhealth_16kHz/" + df['file_id'] + ".wav"
    print("File read")

    # Models initialize ONLY on first row
    # df[['clinicalbert_class', 'clinicalbert_conf']] = df['text'].apply(extract_clinicalbert_outputs)
    # print("Clinicalbert done")
    
    df[['roberta_sentiment', 'roberta_conf']] = df['text'].apply(extract_roberta_outputs)
    print("roberta sentiment done")

    df.to_csv("test_data_for_tools_roberta_clinicalbert.csv", index=False) 
