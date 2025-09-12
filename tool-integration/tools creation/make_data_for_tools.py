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


# ===================================== PROSDIC FEATURES ================================

def calculate_pitch_variability(audio_path):
    # standard deviation of the fundamental frequency (F0) — a measure of how much the pitch varies over time
    """Calculate pitch variability (standard deviation of fundamental frequency) using Librosa"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
        pitches = pitches[pitches > 0]  # Remove invalid pitches
        return np.std(pitches) if len(pitches) > 0 else 0.0
    except Exception as e:
        # print(f"Error in pitch calculation: {e}")
        return 0.0


def calculate_speech_rate(audio_path, sr=16000, frame_length=2048, hop_length=512):
    """Estimate speech rate using energy peak detection"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)   
    # Calculate RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]    
    # Smooth with moving average
    window_size = 5
    rms_smooth = np.convolve(rms, np.ones(window_size)/window_size, mode='same')    
    # Detect peaks (syllable nuclei candidates)
    peaks, _ = find_peaks(rms_smooth, height=np.median(rms_smooth)*1.2, distance=5)    
    # Calculate speaking duration (exclude silence)
    silence_threshold = 0.02 * np.max(rms)
    speech_frames = np.sum(rms > silence_threshold)
    speaking_duration = (speech_frames * hop_length) / sr    
    # Avoid division by zero
    if speaking_duration == 0:
        return 0.0    
    # Syllables per second (speech rate)
    return len(peaks) / speaking_duration


def calculate_pause_frequency(audio_path, silence_threshold=-30, min_pause_duration=0.1):
    """Calculate pause frequency (pauses/second) using Librosa"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop (overlap)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        db = librosa.amplitude_to_db(rms, ref=np.max)

        silence_mask = db < silence_threshold
        silence_changes = np.diff(silence_mask.astype(int))

        pause_starts = np.where(silence_changes == 1)[0]
        pause_ends = np.where(silence_changes == -1)[0]

        # Edge case correction
        if pause_ends.size > 0 and pause_starts.size > 0:
            if pause_ends[0] < pause_starts[0]:
                pause_ends = pause_ends[1:]
            if pause_starts.size > pause_ends.size:
                pause_starts = pause_starts[:-1]

        pause_durations = [(end - start) * hop_length / sr
                           for start, end in zip(pause_starts, pause_ends)
                           if (end - start) * hop_length / sr >= min_pause_duration]

        total_duration = librosa.get_duration(y=y, sr=sr)
        return len(pause_durations) / total_duration if total_duration > 0 else 0.0

    except Exception as e:
        print(f"Error in pause calculation: {e}")
        return 0.0


#====================================== TEXT and VOICE analysis ==============================

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classification_psychbert(text, model_name="./psychbert-finetuned-multiclass"):
    """
    Predicts the mental health category of the input text using a fine-tuned PsychBERT model.
    
    Args:
        text (str): The input text to classify.
        model_name (str): Path or Hugging Face Hub name for the model and tokenizer.
        
    Returns:
        str: The predicted mental health problem category.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get model outputs
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Get predicted class
    predicted_class = logits.argmax(dim=1).item()
    
    # Map class to problem description
    class_map = {
        0: "Negative / unrelated to mental health",
        1: "Mental illnesses",
        2: "Anxiety",
        3: "Depression",
        4: "Social anxiety",
        5: "Loneliness"
    }
    return class_map[predicted_class]
# ------------------------------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F

def load_mentalbert_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load MentalBERT model and tokenizer for inference
    """
    model_name = "mental/mental-bert-base-uncased"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, tokenizer, device

def classify_mental_health_text(text, model, tokenizer, device):
    """
    Perform mental health classification on input text using MentalBERT
    
    Args:
        text (str): Input text to classify
        model: Loaded MentalBERT model
        tokenizer: Loaded MentalBERT tokenizer
        device: Device to run inference on
    
    Returns:
        dict: Classification results with probabilities and predicted class
    """
    # Tokenize input text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits        
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=-1)        
        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=-1).item()        
        # Get confidence score
        confidence = torch.max(probabilities).item()
    
    # Prepare results
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy().flatten().tolist(),
        'raw_logits': logits.cpu().numpy().flatten().tolist()
    }
    
    return results

# Example usage function
def analyze_text_with_mentalbert(text):
    """    Complete workflow to analyze text with MentalBERT"""
    # Load model
    model, tokenizer, device = load_mentalbert_model()
    # Perform classification
    results = classify_mental_health_text(text, model, tokenizer, device)
    return results

def extract_mentalbert_outputs(text):
    results = analyze_text_with_mentalbert(text)
    return pd.Series({
        'mentalbert_depression': results['predicted_class'],
        'mentalbert_conf': results['confidence']
    })
#----------------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def classify_disorder_v01(text):
    """
    Classifies the mental health label for a given text using a pretrained model.
    
    Args:
        text (str): Input transcript text.
        
    Returns:
        str: Predicted label.
    """
    tokenizer = AutoTokenizer.from_pretrained("tahaenesaslanturk/mental-health-classification-v0.1")
    model = AutoModelForSequenceClassification.from_pretrained("tahaenesaslanturk/mental-health-classification-v0.1")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return model.config.id2label[predicted_label]
    
#----------------------------------------------------------------------------------------------------
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, pipeline
)
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalDisorderClassifier:
    """Handles mental disorder classification using Bio_ClinicalBERT and Twitter-RoBERTa models"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.clinical_model = None
        self.clinical_tokenizer = None
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        
    def load_clinical_bert(self):
        """Load Bio_ClinicalBERT model for clinical text classification"""
        try:
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            self.clinical_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.clinical_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.clinical_model.to(self.device)
            self.clinical_model.eval()
            logger.info(f"Clinical BERT model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading Clinical BERT: {e}")
            raise
    
    def load_sentiment_roberta(self):
        """Load Twitter-RoBERTa model for sentiment analysis"""
        try:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()
            logger.info(f"Sentiment RoBERTa model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading Sentiment RoBERTa: {e}")
            raise
    
    def classify_clinical_text(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Classify clinical text using Bio_ClinicalBERT
        
        Args:
            text: Input text to classify
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        if self.clinical_model is None:
            self.load_clinical_bert()
        
        try:
            # Tokenize input
            inputs = self.clinical_tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.clinical_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert to numpy for easier handling
            predictions_np = predictions.cpu().numpy()[0]
            predicted_class = np.argmax(predictions_np)
            confidence = float(predictions_np[predicted_class])
            
            return {
                "predicted_class": int(predicted_class),
                "confidence": confidence,
                "all_scores": predictions_np.tolist(),
                "model_used": "Bio_ClinicalBERT"
            }
            
        except Exception as e:
            logger.error(f"Error in clinical text classification: {e}")
            return {"error": str(e)}
    
    def analyze_sentiment(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Analyze sentiment using Twitter-RoBERTa
        
        Args:
            text: Input text to analyze
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if self.sentiment_model is None:
            self.load_sentiment_roberta()
        
        try:
            # Tokenize input
            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert to numpy
            predictions_np = predictions.cpu().numpy()[0]
            
            # Map to sentiment labels (typical for this model)
            sentiment_labels = ["negative", "neutral", "positive"]
            predicted_sentiment = sentiment_labels[np.argmax(predictions_np)]
            confidence = float(predictions_np[np.argmax(predictions_np)])
            
            return {
                "predicted_sentiment": predicted_sentiment,
                "confidence": confidence,
                "scores": {
                    label: float(score) 
                    for label, score in zip(sentiment_labels, predictions_np)
                },
                "model_used": "Twitter-RoBERTa"
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"error": str(e)}

def run_clinical_bert_classification(text: str, device: str = "auto") -> Dict[str, Any]:
    """
    Standalone function for Bio_ClinicalBERT classification
    
    Args:
        text: Input text to classify
        device: Device to run on ("auto", "cuda", "cpu")
        
    Returns:
        Classification results from Bio_ClinicalBERT
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    classifier = MentalDisorderClassifier(device=device)
    result = classifier.classify_clinical_text(text)
    return result

def extract_clinicalbert_outputs(text):
    results = run_clinical_bert_classification(text)
    return pd.Series({
        'clinicalbert_class': results['predicted_class'],
        'clinicalbert_conf': results['confidence']})

def run_sentiment_roberta_analysis(text: str, device: str = "auto") -> Dict[str, Any]:
    """
    Standalone function for Twitter-RoBERTa sentiment analysis
    
    Args:
        text: Input text to analyze
        device: Device to run on ("auto", "cuda", "cpu")
        
    Returns:
        Sentiment analysis results from Twitter-RoBERTa
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    classifier = MentalDisorderClassifier(device=device)
    result = classifier.analyze_sentiment(text)
    return result

def extract_roberta_outputs(text):
    results = run_sentiment_roberta_analysis(text)
    return pd.Series({
        'roberta_sentiment': results['predicted_sentiment'],
        'roberta_conf': results['confidence']})

#------------------------------------------------------------------------------------------------------
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def analyze_bert_emotion_from_text(text, model_name="boltuix/bert-emotion"):
    """
    Analyze emotion in text using fine-tuned BERT-Emotion model
    
    Args:
        text (str): Input text to analyze
        model_name (str): Hugging Face model identifier (default: "boltuix/bert-emotion")
        
    Returns:
        dict: Emotion analysis result with 'label' and 'score'
    """
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create pipeline
    emotion_analysis = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )
    
    # Analyze and return result
    return emotion_analysis(text)[0]['label']
    
#---------------------------------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelWithLMHead

def get_emotion_t5(text):
  tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
  model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
  output = model.generate(input_ids=input_ids,max_length=2)  
  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]
  
  return label
    
#----------------------------------------------------------------------------------------------------
import torch
import numpy as np
from transformers import Wav2Vec2Model
import torchaudio
from torchaudio.transforms import Resample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", output_hidden_states=True).to(device)

class FineTunedWav2Vec2Model(torch.nn.Module):
    def __init__(self, wav2vec2_model, output_size):
        super(FineTunedWav2Vec2Model, self).__init__()
        self.wav2vec2 = wav2vec2_model
        self.fc = torch.nn.Linear(self.wav2vec2.config.hidden_size, output_size)

    def forward(self, x):
        self.wav2vec2 = self.wav2vec2.double()
        self.fc = self.fc.double()
        outputs = self.wav2vec2(x.double())
        out = outputs.hidden_states[-1]
        out = self.fc(out[:, 0, :])
        return out

def preprocess_audio(audio):
    sample_rate, waveform = audio
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    
    # Normalize audio
    if waveform.dtype != torch.float32:
        waveform = waveform.float() / torch.iinfo(waveform.dtype).max
    
    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform

def predict_emotion_from_voice(audio):
    model_path = "/data/amey_2311cs10/debayan/tools-audmind/SpeechEmotionDetector/model.pth"  # Path to your fine-tuned model
    model = FineTunedWav2Vec2Model(wav2vec2_model, 7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    waveform = preprocess_audio(audio)
    waveform = waveform.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(waveform)

    predicted_label = torch.argmax(output, dim=1).item()
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    return emotion_labels[predicted_label]

def apply_emotion_prediction(row):
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(row['file_id'])
        audio_data = (sample_rate, waveform)
        
        # Predict emotion
        return predict_emotion_from_voice(audio_data)
    except Exception as e:
        # print(f"Error processing {row['file_id']}: {e}")
        return None  # or a default value like "Unknown"


#-========================================== SPECTRAL FEATURES ===========================================
import librosa
import librosa.display
import numpy as np
from scipy import stats

def cpp_tool(audio_path: str, max_quefrency: float = 0.01) -> float:
    """Cepstral Peak Prominence estimation tool"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        spectrum = np.abs(fft.fft(y))
        log_spectrum = np.log(spectrum + 1e-10)
        cepstrum = np.real(fft.ifft(log_spectrum))
        
        # Find peak in quefrency range corresponding to F0 (typically 60-400Hz)
        quefrencies = np.arange(len(cepstrum))/sr
        valid = (quefrencies > 1/400) & (quefrencies < 1/60)
        cpp = np.max(cepstrum[valid]) - np.median(cepstrum[valid])
        return float(cpp)
    except Exception as e:
        print(f"CPP Error: {e}")
        return 0.0
#---------------------------------------------------------------------------------------------------

import numpy as np
import librosa

class LPCAnalyzer:
    def __init__(self, order=12, frame_length=2048, hop_length=512, sr=16000):
        self.order = order
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.sr = sr

    def _levinson_durbin(self, r):
        """Levinson-Durbin recursion"""
        a = np.zeros(self.order + 1)
        a[0] = 1.0
        e = r[0]
        for k in range(1, self.order + 1):
            lam = -np.sum(a[:k] * r[1:k+1][::-1]) / e
            a_new = a.copy()
            a_new[1:k+1] += lam * a[k-1::-1]
            a = a_new
            a[k] = lam
            e *= (1 - lam**2)
        return a[1:], e

    def extract_lpc(self, audio_path):
        """Extract LPC coefficients"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            y = librosa.util.normalize(y)
            frames = librosa.util.frame(y, frame_length=self.frame_length, hop_length=self.hop_length)

            lpc_coeffs = []
            for frame in frames.T:
                frame = frame.copy()
                frame -= np.mean(frame)
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:][:self.order+1]
                if np.sum(np.abs(autocorr)) < 1e-6:
                    lpc_coeffs.append(np.zeros(self.order))
                    continue
                a, _ = self._levinson_durbin(autocorr)
                lpc_coeffs.append(a)

            return np.array(lpc_coeffs)

        except Exception as e:
            print(f"LPC Error ({audio_path}): {str(e)}")
            return np.array([])

    def prediction_error_power(self, lpc_coeffs, audio_path):
        """Calculate prediction error power - indicates vocal tract modeling accuracy
        Higher values may indicate less predictable speech patterns, often associated with depression"""
        y, _ = librosa.load(audio_path, sr=None)
        frames = librosa.util.frame(y, frame_length=self.frame_length, hop_length=self.hop_length)

        total_error_power = 0
        for i, frame in enumerate(frames.T):
            if i < len(lpc_coeffs):
                predicted = np.convolve(frame, lpc_coeffs[i], mode='same')
                error = frame - predicted[:len(frame)]
                total_error_power += np.sum(error**2)

        return total_error_power / len(frames.T)

    def spectral_centroid(self, lpc_coeffs):
        """Calculate average spectral centroid from LPC coefficients
        Lower spectral centroids often correlate with flatter, more monotone speech patterns"""
        centroids = []
        freqs = np.fft.fftfreq(1024, 1 / self.sr)[:512]
        for coeffs in lpc_coeffs:
            h = np.abs(np.fft.fft(np.concatenate([[1], -coeffs]), 1024))[:512]
            centroid = np.sum(freqs * h) / np.sum(h)
            centroids.append(centroid)
        return np.mean(centroids)

    def stability_measure(self, lpc_coeffs):
        """Calculate average pole radius - measure of filter stability
         Values closer to 1.0 indicate less stable vocal tract configurations"""
        pole_radii = []
        for coeffs in lpc_coeffs:
            roots = np.roots(np.concatenate([[1], -coeffs]))
            pole_radii.extend(np.abs(roots))
        return np.mean(pole_radii)

    def first_formant_estimate(self, lpc_coeffs):
        """"Estimate first formant frequency from LPC analysis
        Altered formant patterns are associated with depression-related speech changes"""
        formants = []
        for coeffs in lpc_coeffs:
            roots = np.roots(np.concatenate([[1], -coeffs]))
            angles = np.angle(roots)
            freqs = angles * self.sr / (2 * np.pi)
            valid_freqs = freqs[(freqs > 200) & (freqs < 1000)]
            if len(valid_freqs) > 0:
                formants.append(np.min(valid_freqs))
        return np.mean(formants) if formants else 0

    def get_avg_variability(self, lpc_coeffs):
        """
        Calculate the average variability of LPC coefficients.
    
        Parameters:
        - lpc_coeffs (np.ndarray): 2D array (frames x order) of LPC coefficients
    
        Returns:
        - float: Average variability (mean of frame-wise standard deviations)
        """
        # Compute standard deviation across coefficients for each frame
        frame_variability = np.std(lpc_coeffs, axis=1)
        # Return the mean variability
        return np.mean(frame_variability)

def apply_lpc_analysis(row, analyzer):
        audio_path = row["file_id"]
        lpc_coeffs = analyzer.extract_lpc(audio_path)    
        if lpc_coeffs.size == 0:
            return pd.Series({
                "lpc_error_power": None,
                "lpc_spectral_centroid": None,
                "lpc_stability": None,
                "lpc_first_formant": None,
                "lpc_avg_variability": None})    
        return pd.Series({
            "lpc_error_power": analyzer.prediction_error_power(lpc_coeffs, audio_path),
            "lpc_spectral_centroid": analyzer.spectral_centroid(lpc_coeffs),
            "lpc_stability": analyzer.stability_measure(lpc_coeffs),
            "lpc_first_formant": analyzer.first_formant_estimate(lpc_coeffs),
            "lpc_avg_variability": analyzer.get_avg_variability(lpc_coeffs)})

#---------------------------------------------------------------------------------------------------

import librosa
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MFCCMentalHealthAnalyzer:
    """
    Comprehensive MFCC feature extraction class for mental health analysis.
    All methods return scalar values or strings for DataFrame storage.
    """
    
    def __init__(self, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_all_features(self, audio_path):
        """
        Extract all MFCC-based mental health features from audio file.
        
        Returns:
        dict: Dictionary containing all scalar/string features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                       n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Extract delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Calculate all features
            features = {
                # Core MFCC features
                'mfcc2_mean': self.get_mfcc2_mean(mfccs),
                'mfcc_overall_variability': self.get_overall_variability(mfccs),
                'depression_risk_indicator': self.get_depression_risk_indicator(mfccs),
                
                # Additional important features
                'mfcc_spectral_centroid': self.get_mfcc_spectral_centroid(mfccs, sr),
                'mfcc_energy_concentration': self.get_energy_concentration(mfccs),
                'mfcc_dynamic_range': self.get_dynamic_range(mfccs),
                'mfcc_coefficient_stability': self.get_coefficient_stability(mfccs),
                'mfcc_first_formant_estimate': self.get_first_formant_estimate(mfccs, sr),
                
                # Delta-based features
                'delta_mfcc_variability': self.get_delta_variability(delta_mfccs),
                'delta2_mfcc_energy': self.get_delta2_energy(delta2_mfccs),
                
                # Advanced features
                'mfcc_entropy': self.get_mfcc_entropy(mfccs),
                'mfcc_kurtosis': self.get_mfcc_kurtosis(mfccs),
                'mfcc_zero_crossing_analog': self.get_zero_crossing_analog(mfccs),
                'mfcc_spectral_rolloff_estimate': self.get_spectral_rolloff_estimate(mfccs),
                'mfcc_bandwidth_estimate': self.get_bandwidth_estimate(mfccs),
                
                # Clinical interpretation
                'mental_health_risk_level': self.get_mental_health_risk_level(mfccs),
                'speech_monotony_indicator': self.get_speech_monotony_indicator(mfccs),
                'vocal_tract_stability': self.get_vocal_tract_stability(mfccs)
            }
            
            return features
            
        except Exception as e:
            print(f"Error in MFCC feature extraction: {str(e)}")
            return self._get_default_features()
    
    def get_mfcc2_mean(self, mfccs):
        """
        Calculate MFCC2 mean value - critical for depression detection.
        Lower values (< -2.0) often indicate depression.
        """
        return float(np.mean(mfccs[1, :]))  # MFCC2 is index 1
    
    def get_overall_variability(self, mfccs):
        """
        Calculate overall MFCC variability across all coefficients.
        Lower values indicate more monotone speech patterns.
        """
        return float(np.mean(np.std(mfccs, axis=1)))
    
    def get_depression_risk_indicator(self, mfccs):
        """
        Depression risk indicator based on MFCC2 threshold analysis.
        Returns: 'HIGH', 'MODERATE', or 'LOW'
        """
        mfcc2_mean = np.mean(mfccs[1, :])
        
        if mfcc2_mean < -2.5:
            return 'HIGH'
        elif mfcc2_mean < -1.5:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def get_mfcc_spectral_centroid(self, mfccs, sr):
        """
        Estimate spectral centroid from MFCC coefficients.
        Lower values indicate flatter, more monotone speech.
        """
        # Approximate spectral centroid from MFCC distribution
        freq_weights = np.arange(len(mfccs)) * (sr / 2) / len(mfccs)
        mean_coeffs = np.mean(np.abs(mfccs), axis=1)
        
        if np.sum(mean_coeffs) > 0:
            centroid = np.sum(freq_weights * mean_coeffs) / np.sum(mean_coeffs)
            return float(centroid)
        else:
            return 0.0
    
    def get_energy_concentration(self, mfccs):
        """
        Calculate energy concentration in lower frequency bands.
        Higher values indicate more energy in lower frequencies (depression marker).
        """
        lower_band_energy = np.mean(np.abs(mfccs[:4, :]))  # First 4 coefficients
        total_energy = np.mean(np.abs(mfccs))
        
        if total_energy > 0:
            return float(lower_band_energy / total_energy)
        else:
            return 0.0
    
    def get_dynamic_range(self, mfccs):
        """
        Calculate dynamic range of MFCC coefficients.
        Lower values indicate less dynamic speech.
        """
        return float(np.max(mfccs) - np.min(mfccs))
    
    def get_coefficient_stability(self, mfccs):
        """
        Measure how stable the MFCC coefficients are over time.
        Higher values indicate more stable (potentially monotone) speech.
        """
        temporal_stability = []
        for i in range(len(mfccs)):
            coeff_var = np.var(mfccs[i, :])
            temporal_stability.append(coeff_var)
        
        # Return inverse of mean variance (higher = more stable)
        mean_variance = np.mean(temporal_stability)
        if mean_variance > 0:
            return float(1.0 / (1.0 + mean_variance))
        else:
            return 1.0
    
    def get_first_formant_estimate(self, mfccs, sr):
        """
        Estimate first formant frequency from MFCC coefficients.
        Abnormal values may indicate altered vocal tract configuration.
        """
        # Rough approximation of F1 from MFCC2 and MFCC3
        mfcc2_mean = np.mean(mfccs[1, :])
        mfcc3_mean = np.mean(mfccs[2, :])
        
        # Empirical formula for F1 estimation
        f1_estimate = 600 + (mfcc2_mean * 100) + (mfcc3_mean * 50)
        return float(max(200, min(1000, f1_estimate)))  # Clamp to reasonable range
    
    def get_delta_variability(self, delta_mfccs):
        """
        Calculate variability in delta MFCC coefficients.
        Lower values indicate less dynamic speech changes.
        """
        return float(np.mean(np.std(delta_mfccs, axis=1)))
    
    def get_delta2_energy(self, delta2_mfccs):
        """
        Calculate energy in delta-delta coefficients.
        Lower values indicate less acceleration in speech changes.
        """
        return float(np.mean(np.abs(delta2_mfccs)))
    
    def get_mfcc_entropy(self, mfccs):
        """
        Calculate entropy of MFCC coefficient distribution.
        Lower values indicate less information content (monotone speech).
        """
        # Normalize coefficients to probability distribution
        abs_coeffs = np.abs(mfccs.flatten())
        abs_coeffs = abs_coeffs / np.sum(abs_coeffs) if np.sum(abs_coeffs) > 0 else abs_coeffs
        
        # Calculate entropy
        entropy = -np.sum(abs_coeffs * np.log2(abs_coeffs + 1e-10))
        return float(entropy)
    
    def get_mfcc_kurtosis(self, mfccs):
        """
        Calculate kurtosis of MFCC coefficients.
        Higher values indicate more peaked distributions (less variability).
        """
        flattened = mfccs.flatten()
        return float(stats.kurtosis(flattened))
    
    def get_zero_crossing_analog(self, mfccs):
        """
        MFCC analog of zero-crossing rate.
        Count sign changes in MFCC coefficients over time.
        """
        sign_changes = 0
        total_comparisons = 0
        
        for i in range(len(mfccs)):
            coeff_series = mfccs[i, :]
            for j in range(len(coeff_series) - 1):
                if coeff_series[j] * coeff_series[j + 1] < 0:
                    sign_changes += 1
                total_comparisons += 1
        
        return float(sign_changes / total_comparisons) if total_comparisons > 0 else 0.0
    
    def get_spectral_rolloff_estimate(self, mfccs):
        """
        Estimate spectral rolloff frequency from MFCC distribution.
        Lower values indicate more energy in lower frequencies.
        """
        # Find coefficient index where 85% of energy is contained
        abs_coeffs = np.mean(np.abs(mfccs), axis=1)
        cumulative_energy = np.cumsum(abs_coeffs)
        total_energy = cumulative_energy[-1]
        
        if total_energy > 0:
            rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
            if len(rolloff_idx) > 0:
                return float(rolloff_idx[0] / len(mfccs))
        
        return 0.85  # Default rolloff ratio
    
    def get_bandwidth_estimate(self, mfccs):
        """
        Estimate spectral bandwidth from MFCC coefficient spread.
        Lower values indicate narrower frequency content.
        """
        mean_coeffs = np.mean(np.abs(mfccs), axis=1)
        
        # Calculate weighted standard deviation
        weights = np.arange(len(mean_coeffs))
        if np.sum(mean_coeffs) > 0:
            weighted_mean = np.sum(weights * mean_coeffs) / np.sum(mean_coeffs)
            weighted_var = np.sum(mean_coeffs * (weights - weighted_mean) ** 2) / np.sum(mean_coeffs)
            return float(np.sqrt(weighted_var))
        else:
            return 0.0
    
    def get_mental_health_risk_level(self, mfccs):
        """
        Overall mental health risk assessment based on multiple MFCC indicators.
        Returns: 'HIGH', 'MODERATE', or 'LOW'
        """
        # Multiple risk factors
        mfcc2_mean = np.mean(mfccs[1, :])
        variability = np.mean(np.std(mfccs, axis=1))
        energy_concentration = self.get_energy_concentration(mfccs)
        
        risk_score = 0
        
        # MFCC2 factor
        if mfcc2_mean < -2.5:
            risk_score += 2
        elif mfcc2_mean < -1.5:
            risk_score += 1
        
        # Variability factor
        if variability < 0.3:
            risk_score += 2
        elif variability < 0.5:
            risk_score += 1
        
        # Energy concentration factor
        if energy_concentration > 0.7:
            risk_score += 1
        
        if risk_score >= 4:
            return 'HIGH'
        elif risk_score >= 2:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def get_speech_monotony_indicator(self, mfccs):
        """
        Indicator of speech monotony based on coefficient variation.
        Returns: 'HIGH', 'MODERATE', or 'LOW'
        """
        variability = np.mean(np.std(mfccs, axis=1))
        
        if variability < 0.3:
            return 'HIGH'
        elif variability < 0.6:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def get_vocal_tract_stability(self, mfccs):
        """
        Assessment of vocal tract configuration stability.
        Returns: 'STABLE', 'MODERATE', or 'VARIABLE'
        """
        stability = self.get_coefficient_stability(mfccs)
        
        if stability > 0.8:
            return 'STABLE'
        elif stability > 0.6:
            return 'MODERATE'
        else:
            return 'VARIABLE'
    
    def _get_default_features(self):
        """Return default feature values in case of error."""
        return {
            'mfcc2_mean': 0.0,
            'mfcc_overall_variability': 0.0,
            'depression_risk_indicator': 'UNKNOWN',
            'mfcc_spectral_centroid': 0.0,
            'mfcc_energy_concentration': 0.0,
            'mfcc_dynamic_range': 0.0,
            'mfcc_coefficient_stability': 0.0,
            'mfcc_first_formant_estimate': 0.0,
            'delta_mfcc_variability': 0.0,
            'delta2_mfcc_energy': 0.0,
            'mfcc_entropy': 0.0,
            'mfcc_kurtosis': 0.0,
            'mfcc_zero_crossing_analog': 0.0,
            'mfcc_spectral_rolloff_estimate': 0.0,
            'mfcc_bandwidth_estimate': 0.0,
            'mental_health_risk_level': 'UNKNOWN',
            'speech_monotony_indicator': 'UNKNOWN',
            'vocal_tract_stability': 'UNKNOWN'
        }

def extract_mfcc_features(row, analyzer):
        # audio_path = row["file_id"]
        # return pd.Series(analyzer.extract_all_features(audio_path))
    try:
        return pd.Series(analyzer.extract_all_features(row["file_id"]))
    except Exception as e:
        # print(f"MFCC error on {row['file_id']}: {e}")
        return pd.Series(analyzer._get_default_features())

#-=======================================================================================================

if __name__ == "__main__":
    df = pd.read_csv("/data/amey_2311cs10/debayan/train_audmind.csv")
    df['file_id'] = "/data/amey_2311cs10/debayan/train_mentalhealth_16kHz/" + df['file_id'] + ".wav"
    
    df['Pitch Variability'] = df['file_id'].apply(calculate_pitch_variability)
    df['Speech Rate'] = df['file_id'].apply(calculate_speech_rate)
    df['Pause Frequency'] = df['file_id'].apply(calculate_pause_frequency)

    df['pyschbert_classification'] = df['text'].apply(classification_psychbert)
    df[['mentalbert_depression', 'mentalbert_conf']] = df['text'].apply(extract_mentalbert_outputs)
    df['v01_classification'] = df['text'].apply(classify_disorder_v01)
    
    df[['clinicalbert_class', 'clinicalbert_conf']] = df['text'].apply(extract_clinicalbert_outputs)
    df[['roberta_sentiment', 'roberta_conf']] = df['text'].apply(extract_roberta_outputs)
    
    df['bertemotion_classification'] = df['text'].apply(analyze_bert_emotion_from_text)
    df['t5_emotion'] = df['text'].apply(get_emotion_t5)
    df['t5_emotion'] = df['t5_emotion'].apply(lambda x: x.replace("<pad>", "").strip())
    df['wav2vec2_emotion_voice'] = df.apply(apply_emotion_prediction, axis=1)

    df['cpp values'] = df['file_id'].apply(cpp_tool)

    lpc_analyzer = LPCAnalyzer()    
    df[["lpc_error_power", "lpc_spectral_centroid",  "lpc_stability", "lpc_first_formant", "lpc_avg_variability"]] = df.apply(lambda row: apply_lpc_analysis(row, lpc_analyzer), axis=1)

    mfcc_analyzer = MFCCMentalHealthAnalyzer()
    df = df.join(df.apply(lambda row: extract_mfcc_features(row, mfcc_analyzer), axis=1))


    
    df.to_csv("training_data_for_tools.csv", index=False) 

