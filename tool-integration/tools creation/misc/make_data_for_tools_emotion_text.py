import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
from typing import Dict, Any

# 1. BERT Emotion Classifier Manager
class BertEmotionManager:
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
                model_name = "bhadresh-savani/bert-base-uncased-emotion"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"BertEmotionManager init error: {e}")
                raise
    
    def analyze_emotion(self, text: str) -> str:
        self.initialize()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        return emotion_labels[torch.argmax(probabilities).item()]

# 2. T5 Emotion Classifier Manager
class T5EmotionManager:
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
                model_name = "mrm8488/t5-base-finetuned-emotion"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"T5EmotionManager init error: {e}")
                raise
    
    def get_emotion(self, text: str) -> str:
        self.initialize()
        input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Wav2Vec2 Emotion Classifier Manager (Handles multimodal input)
# class Wav2Vec2EmotionManager:
#     _instance = None
    
#     def __new__(cls, device="auto"):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
#             cls._instance.audio_model = None
#             cls._instance.text_model = None
#         return cls._instance
    
#     def initialize(self):
#         if self.audio_model is None:
#             try:
#                 # Audio processing model
#                 self.audio_model = pipeline(
#                     "audio-classification", 
#                     model="superb/wav2vec2-base-superb-ks",
#                     device=self.device
#                 )
                
#                 # Text processing model
#                 self.text_model = pipeline(
#                     "text-classification",
#                     model="j-hartmann/emotion-english-distilroberta-base",
#                     device=self.device
#                 )
#             except Exception as e:
#                 print(f"Wav2Vec2EmotionManager init error: {e}")
#                 raise
    
#     def predict_emotion(self, row: Dict[str, Any]) -> str:
#         self.initialize()
#         audio_path = row['file_id']
#         text = row['text']
        
#         # Process audio
#         audio_result = self.audio_model(audio_path)[0]['label']
        
#         # Process text
#         text_result = self.text_model(text)[0]['label']
        
#         # Fusion logic (simplified example)
#         return f"{audio_result}_{text_result}"

class Wav2Vec2EmotionManager:
    _instance = None
    
    def __new__(cls, device="auto"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.feature_extractor = None
            cls._instance.model = None
        return cls._instance
    
    def initialize(self):
        if self.model is None:
            try:
                model_name = "facebook/wav2vec2-base-960h"
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Wav2Vec2EmotionManager init error: {e}")
                raise
    
    def predict_emotion(self, row: Dict[str, Any]) -> str:
        self.initialize()
        audio_path = row['file_id']
        
        # Load audio with librosa
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Apply truncation and padding
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=True,           # Enable truncation
            max_length=320000,         # ~20 seconds at 16kHz
            padding="max_length"       # Pad to max length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Return predicted emotion
        emotion_labels = ['angry', 'happy', 'sad', 'neutral']  # Adjust based on your model
        return emotion_labels[torch.argmax(predictions).item()]



# 4. DataFrame Processing Functions
def analyze_bert_emotion_from_text(text: str) -> str:
    manager = BertEmotionManager()
    return manager.analyze_emotion(text)

def get_emotion_t5(text: str) -> str:
    manager = T5EmotionManager()
    emotion = manager.get_emotion(text)
    return emotion.replace("<pad>", "").strip()

def apply_emotion_prediction(row: pd.Series) -> str:
    manager = Wav2Vec2EmotionManager()
    return manager.predict_emotion(row.to_dict())


if __name__ == "__main__":
    df = pd.read_csv("/data/amey_2311cs10/debayan/test_audmind.csv")
    df['file_id'] = "/data/amey_2311cs10/debayan/test_mentalhealth_16kHz/" + df['file_id'] + ".wav"
    print("File read")
    
    # Optimized emotion classification calls
    df['bertemotion_classification'] = df['text'].apply(analyze_bert_emotion_from_text)
    df['t5_emotion'] = df['text'].apply(get_emotion_t5)
    print("Text Analysis done")
    # df['wav2vec2_emotion_voice'] = df.apply(apply_emotion_prediction, axis=1)
    # print("Voice Analysis done")
    
    df.to_csv("test_data_for_tools_emotion_text.csv", index=False)
    print("DONE")
