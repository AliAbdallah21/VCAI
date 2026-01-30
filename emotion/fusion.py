# fusion.py

import numpy as np
from .voice_emotion import EmotionalAgent
from .text_emotion import TextEmotionAnalyzer

class EmotionFusion:
    """
    Fuses voice and text emotion predictions
    Combines both modalities for better accuracy
    """
    
    def __init__(self, voice_model_path):
        self.voice_agent = EmotionalAgent(voice_model_path)
        self.text_analyzer = TextEmotionAnalyzer()
        
    def fuse_predictions(self, voice_result, text_result, voice_weight=0.6, text_weight=0.4):
        """
        Combine voice and text predictions
        
        Args:
            voice_result (dict): Voice emotion prediction
            text_result (dict): Text emotion prediction
            voice_weight (float): Weight for voice prediction
            text_weight (float): Weight for text prediction
            
        Returns:
            dict: Fused emotion prediction
        """
        # Map emotions to common labels
        emotion_map = {
            'angry': 'angry',
            'happy': 'happy',
            'sad': 'hesitant',
            'fear': 'hesitant',
            'surprise': 'interested',
            'neutral': 'neutral'
        }
        
        # Combine probabilities
        voice_probs = voice_result['all_probabilities']
        text_probs = text_result['all_probabilities']
        
        fused_probs = {}
        for emotion in voice_probs.keys():
            mapped_text_emotion = emotion_map.get(emotion, emotion)
            text_prob = text_probs.get(mapped_text_emotion, 0)
            
            fused_probs[emotion] = (
                voice_weight * voice_probs[emotion] + 
                text_weight * text_prob
            )
        
        # Get final prediction
        final_emotion = max(fused_probs, key=fused_probs.get)
        final_confidence = fused_probs[final_emotion]
        
        return {
            'emotion': final_emotion,
            'confidence': final_confidence,
            'all_probabilities': fused_probs,
            'source': 'fusion',
            'voice_prediction': voice_result['emotion'],
            'text_prediction': text_result['emotion']
        }
    
    def predict_from_audio_and_text(self, audio_path, text):
        """
        Predict emotion from both audio and text
        
        Args:
            audio_path (str): Path to audio file
            text (str): Transcript or text
            
        Returns:
            dict: Fused emotion prediction
        """
        voice_result = self.voice_agent.predict_emotion(audio_path)
        text_result = self.text_analyzer.predict_emotion(text)
        
        return self.fuse_predictions(voice_result, text_result)