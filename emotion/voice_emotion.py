# voice_emotion.py

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EmotionalAgent:
    """
    Emotional Agent for recognizing emotions from audio files.
    Integrates your trained Wav2Vec2 model into the VCAI system.
    """
    
    def __init__(self, model_path, device=None):
        """Initialize the Emotional Agent"""
        self.model_path = Path(model_path)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Emotion labels
        self.emotion_labels = {
            0: "angry",
            1: "happy",
            2: "hesitant",
            3: "interested",
            4: "neutral"
        }
        
        self.label_to_id = {v: k for k, v in self.emotion_labels.items()}
        
        print(f"🤖 Loading Emotional Agent...")
        print(f"📍 Model path: {self.model_path}")
        print(f"💻 Device: {self.device}")
        
        self.model = self.load_model()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )
        
        print(f"✅ Emotional Agent loaded successfully!")
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(self.emotion_labels),
                ignore_mismatched_sizes=True
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise Exception(f"❌ Failed to load model: {e}")
    
    def load_audio(self, audio_path, target_sr=16000, max_length=5):
        """Load and preprocess audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            max_samples = max_length * target_sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                audio = np.pad(audio, (0, max_samples - len(audio)))
            return audio
        except Exception as e:
            raise Exception(f"❌ Failed to load audio: {e}")
    
    def predict_emotion(self, audio_path):
        """
        Predict emotion from audio file
        
        Returns:
            dict: {
                'emotion': str,
                'confidence': float,
                'all_probabilities': dict
            }
        """
        try:
            # Load audio
            audio = self.load_audio(audio_path)
            
            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt"
            )
            
            input_values = inputs.input_values.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(input_values).logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
            
            # Get all probabilities
            all_probs = {}
            for idx, emotion in self.emotion_labels.items():
                all_probs[emotion] = probabilities[0][idx].item()
            
            predicted_emotion = self.emotion_labels[predicted_id]
            
            return {
                'emotion': predicted_emotion,
                'confidence': confidence,
                'all_probabilities': all_probs
            }
            
        except Exception as e:
            raise Exception(f"❌ Prediction failed: {e}")
    
    def predict_emotion_from_array(self, audio_array, sample_rate=16000):
        """Predict emotion from audio numpy array (for real-time audio)"""
        try:
            # Resample if needed
            if sample_rate != 16000:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
            
            # Truncate or pad
            max_samples = 5 * 16000
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]
            else:
                audio_array = np.pad(audio_array, (0, max_samples - len(audio_array)))
            
            # Extract features
            inputs = self.feature_extractor(
                audio_array,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt"
            )
            
            input_values = inputs.input_values.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                logits = self.model(input_values).logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_id = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_id].item()
            
            # Get all probabilities
            all_probs = {}
            for idx, emotion in self.emotion_labels.items():
                all_probs[emotion] = probabilities[0][idx].item()
            
            predicted_emotion = self.emotion_labels[predicted_id]
            
            return {
                'emotion': predicted_emotion,
                'confidence': confidence,
                'all_probabilities': all_probs
            }
            
        except Exception as e:
            raise Exception(f"❌ Prediction from array failed: {e}")