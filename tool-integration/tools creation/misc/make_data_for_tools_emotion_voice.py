import os
import warnings
import logging
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

# Suppress excessive warnings (optional)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")

MODEL_DIR = "/data/amey_2311cs10/debayan/tools-audmind/SpeechEmotionDetector"
CHECKPOINT = os.path.join(MODEL_DIR, "model.pth")
CPU_CORES = max(1, mp.cpu_count() - 1)           # Leave 1 core free
TARGET_SR = 16_000                               # Wav2Vec2 expects 16 kHz[3]
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy",
            "Neutral", "Sad", "Surprise"]        # Index ↦ label

# Memory optimization settings
torch.cuda.set_per_process_memory_fraction(0.5)  # Use 50% of GPU memory[3]
torch.cuda.empty_cache()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------
from transformers import Wav2Vec2Model

# 3.1  Load base Wav2Vec2 backbone (once)
base_wav2vec2 = (
    Wav2Vec2Model
    .from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
    .to(DEVICE)
    .eval()
)

# 3.2  Wrap it with your classification head
class FineTunedWav2Vec2Model(torch.nn.Module):
    def __init__(self, wav2vec2_model, output_size: int = 7):
        super().__init__()
        self.wav2vec2 = wav2vec2_model
        self.fc = torch.nn.Linear(self.wav2vec2.config.hidden_size, output_size)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.wav2vec2(x).hidden_states[-1]  # [B, T, H]
        logits = self.fc(hidden[:, 0, :])            # cls-token style pooling
        return logits

# 3.3  Instantiate and load weights only ONCE
emotion_model = FineTunedWav2Vec2Model(base_wav2vec2, len(EMOTIONS)).to(DEVICE)
emotion_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
emotion_model.eval()
logging.info("Fine-tuned Wav2Vec2 model loaded.")

#---------------------------------------------------------

def preprocess_waveform(waveform: torch.Tensor,
                        sample_rate: int,
                        target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Mono-ize, normalize, and resample a raw waveform.
    Returns a 1-D float32 tensor on CPU.
    """
    if waveform.ndim == 2:          # stereo → mono
        waveform = waveform.mean(dim=0)
    if sample_rate != target_sr:     # resample if needed
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    # torchaudio.load(norm=True) already returns float32 in [-1,1][4]
    return waveform


def _predict_emotion(file_path: str) -> str:
    """
    Child-safe function so that each multiprocessing worker:
    1. Loads the file.
    2. Forwards it through the GLOBAL model (read-only).
    3. Returns the predicted string label.
    """
    try:
        waveform, sr = torchaudio.load(file_path)
        waveform = preprocess_waveform(waveform, sr)          # CPU
        waveform = waveform.unsqueeze(0).to(DEVICE)           # [1, T]
        logits = emotion_model(waveform)                      # GPU/CPU
        pred_idx = torch.argmax(logits, dim=1).item()
        return EMOTIONS[pred_idx]
    except Exception as err:
        logging.error(f"{file_path}: {err}")
        return "Unknown"                                      # graceful fallback


def batch_infer_emotions(file_series: pd.Series,
                         n_jobs: int = CPU_CORES,
                         show_progress: bool = True) -> pd.Series:
    """
    Parallel inference over a Series of file paths.
    Uses multiprocessing Pool + tqdm for a live progress bar.
    """
    file_paths = file_series.tolist()
    logging.info(f"Starting inference on {len(file_paths):,} audio files "
                 f"using {n_jobs} processes and device={DEVICE}.")
    
    # Partial function needed because Pool.map only accepts one arg
    with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
        mapper = pool.imap(_predict_emotion, file_paths, chunksize=64)
        if show_progress:
            mapper = tqdm(mapper, total=len(file_paths), desc="Inferring")
        results = list(mapper)
    return pd.Series(results, index=file_series.index, dtype="object")

#----------------------------------------------------------------------------

# if __name__ == "__main__":

#     # 7.1  Load CSV and construct absolute paths
#     df = pd.read_csv("/data/amey_2311cs10/debayan/train_audmind.csv")
#     df["file_id"] = "/data/amey_2311cs10/debayan/train_mentalhealth_16kHz/" + df["file_id"].astype(str) + ".wav"
    
#     # 7.2  **Parallel emotion inference** in one line
#     df["wav2vec2_emotion_voice"] = batch_infer_emotions(df["file_id"])
    
#     # 7.3  Optionally save intermediate results
#     df.to_csv("training_data_with_emotions_voice.csv", index=False)
#     logging.info("Finished inference and saved new CSV.")


def optimized_emotion_prediction(file_series: pd.Series) -> pd.Series:
    """
    Memory-optimized sequential processing for emotion prediction
    """
    results = []
    
    for file_path in tqdm(file_series, desc="Processing audio files"):
        try:
            # Load and process audio
            waveform, sr = torchaudio.load(file_path)
            waveform = preprocess_waveform(waveform, sr)
            waveform = waveform.unsqueeze(0).to(DEVICE)
            
            # Inference with memory management
            with torch.no_grad():
                logits = emotion_model(waveform)
                pred_idx = torch.argmax(logits, dim=1).item()
                results.append(EMOTIONS[pred_idx])
            
            # Clear variables and GPU cache
            del waveform, logits
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            results.append("Unknown")
    
    return pd.Series(results, index=file_series.index)

# In your main function:
if __name__ == "__main__":
    df = pd.read_csv("/data/amey_2311cs10/debayan/train_audmind.csv")
    df["file_id"] = (
        "/data/amey_2311cs10/debayan/train_mentalhealth_16kHz/" +
        df["file_id"].astype(str) + ".wav"
    )
    
    # Use optimized sequential processing
    df["wav2vec2_emotion_voice"] = optimized_emotion_prediction(df["file_id"])
    
    df.to_csv("training_data_with_emotions_voice.csv", index=False)
    logging.info("Processing completed successfully")


# import torch
# import torchaudio
# from torchaudio.transforms import Resample
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import logging
# import gc
# from transformers import Wav2Vec2Model

# # Configuration
# MODEL_DIR = "/data/amey_2311cs10/debayan/tools-audmind/SpeechEmotionDetector"
# CHECKPOINT = "/data/amey_2311cs10/debayan/tools-audmind/SpeechEmotionDetector/model.pth"
# TARGET_SR = 16000
# EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Memory optimization settings
# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.4)  # Use only 40% of GPU memory[6]
#     torch.cuda.empty_cache()

# logging.basicConfig(level=logging.INFO)

# # Load model once (singleton pattern)
# class Wav2Vec2EmotionManager:
#     _instance = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance.model = None
#             cls._instance.base_model = None
#         return cls._instance
    
#     def initialize(self):
#         if self.model is None:
#             # Load base Wav2Vec2 model
#             self.base_model = (
#                 Wav2Vec2Model
#                 .from_pretrained("facebook/wav2vec2-base", output_hidden_states=True)
#                 .to(DEVICE)
#                 .eval()
#             )
            
#             # Load fine-tuned model
#             self.model = FineTunedWav2Vec2Model(self.base_model, len(EMOTIONS)).to(DEVICE)
#             self.model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
#             self.model.eval()
#             logging.info(f"Wav2Vec2 emotion model loaded on {DEVICE}")

# class FineTunedWav2Vec2Model(torch.nn.Module):
#     def __init__(self, wav2vec2_model, output_size):
#         super(FineTunedWav2Vec2Model, self).__init__()
#         self.wav2vec2 = wav2vec2_model
#         self.fc = torch.nn.Linear(self.wav2vec2.config.hidden_size, output_size)

#     def forward(self, x):
#         self.wav2vec2 = self.wav2vec2.double()
#         self.fc = self.fc.double()
#         outputs = self.wav2vec2(x.double())
#         out = outputs.hidden_states[-1]
#         out = self.fc(out[:, 0, :])
#         return out

# def preprocess_audio(waveform, sample_rate):
#     """Preprocess audio with memory optimization"""
#     if isinstance(waveform, np.ndarray):
#         waveform = torch.from_numpy(waveform)
#     if waveform.dim() == 2:
#         waveform = waveform.mean(dim=0)
    
#     # Normalize audio
#     if waveform.dtype != torch.float32:
#         waveform = waveform.float() / torch.iinfo(waveform.dtype).max
    
#     # Resample to 16kHz if needed
#     if sample_rate != TARGET_SR:
#         resampler = Resample(orig_freq=sample_rate, new_freq=TARGET_SR)
#         waveform = resampler(waveform)
    
#     return waveform

# def process_batch_emotions(file_paths, batch_size=2):
#     """
#     Process audio files in small batches to manage memory efficiently
    
#     Args:
#         file_paths: List of audio file paths
#         batch_size: Number of files to process at once (recommended: 1-4 for memory safety)
    
#     Returns:
#         List of emotion predictions
#     """
#     manager = Wav2Vec2EmotionManager()
#     manager.initialize()
    
#     all_results = []
    
#     # Process in small batches
#     for i in tqdm(range(0, len(file_paths), batch_size), desc=f"Processing (batch_size={batch_size})"):
#         batch_files = file_paths[i:i + batch_size]
#         batch_results = []
        
#         for file_path in batch_files:
#             try:
#                 # Load and preprocess audio
#                 waveform, sample_rate = torchaudio.load(file_path)
#                 waveform = preprocess_audio(waveform, sample_rate)
#                 waveform = waveform.unsqueeze(0).to(DEVICE)
                
#                 # Predict emotion with memory management
#                 with torch.no_grad():
#                     output = manager.model(waveform)
#                     predicted_label = torch.argmax(output, dim=1).item()
#                     batch_results.append(EMOTIONS[predicted_label])
                
#                 # Clean up intermediate tensors
#                 del waveform, output
                
#             except Exception as e:
#                 logging.error(f"Error processing {file_path}: {e}")
#                 batch_results.append("Unknown")
        
#         all_results.extend(batch_results)
        
#         # Clear GPU memory after each batch
#         if DEVICE.type == 'cuda':
#             torch.cuda.empty_cache()
#         gc.collect()  # Force Python garbage collection[3]
    
#     return all_results

# def apply_emotion_prediction_batch(df, batch_size=2):
#     """
#     Apply emotion prediction to DataFrame with configurable batch size
    
#     Args:
#         df: DataFrame with 'file_id' column
#         batch_size: Batch size for processing (start with 1-2 for safety)
        
#     Returns:
#         Series with emotion predictions
#     """
#     file_paths = df['file_id'].tolist()
#     results = process_batch_emotions(file_paths, batch_size=batch_size)
#     return pd.Series(results, index=df.index)

# # Updated main function
# if __name__ == "__main__":
#     df = pd.read_csv("/data/amey_2311cs10/debayan/train_audmind.csv")
#     df['file_id'] = "/data/amey_2311cs10/debayan/train_mentalhealth_16kHz/" + df['file_id'] + ".wav"
    
#     # Start with very small batch size for safety
#     # You can increase gradually: 1 -> 2 -> 4 -> 8 if memory allows
#     BATCH_SIZE = 4  # Start here, increase if no memory errors
    
#     logging.info(f"Starting emotion prediction with batch_size={BATCH_SIZE}")
    
#     # Process with small batches
#     df['wav2vec2_emotion_voice'] = apply_emotion_prediction_batch(df, batch_size=BATCH_SIZE)
    
#     # Save results
#     df.to_csv("training_data_with_emotions_voice.csv", index=False)
#     logging.info("Processing completed successfully")
