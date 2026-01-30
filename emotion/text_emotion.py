# text_emotion.py

from transformers import pipeline
import torch

class TextEmotionAnalyzer:
    """
    Analyze emotions from text
    Useful for analyzing transcripts or chat messages
    """
    
    def __init__(self):
        # Load text emotion model
        self.model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
    def predict_emotion(self, text):
        """
        Predict emotion from text
        
        Args:
            text (str): Input text
            
        Returns:
            dict: {
                'emotion': str,
                'confidence': float,
                'all_probabilities': dict
            }
        """
        results = self.model(text)[0]
        
        # Get top emotion
        top_emotion = max(results, key=lambda x: x['score'])
        
        # Format results
        all_probs = {item['label']: item['score'] for item in results}
        
        return {
            'emotion': top_emotion['label'],
            'confidence': top_emotion['score'],
            'all_probabilities': all_probs
        }
    
    def analyze_transcript(self, transcript_segments):
        """
        Analyze emotion across transcript segments
        
        Args:
            transcript_segments (list): List of text segments
            
        Returns:
            list: Emotion predictions for each segment
        """
        results = []
        for segment in transcript_segments:
            emotion = self.predict_emotion(segment)
            results.append(emotion)
        return results