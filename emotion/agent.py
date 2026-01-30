# agent.py

from .voice_emotion import EmotionalAgent
from .text_emotion import TextEmotionAnalyzer
from .fusion import EmotionFusion
from .config import EmotionConfig

class EmotionAgent:
    """
    Main Emotion Agent for VCAI
    Orchestrates all emotion recognition capabilities
    """
    
    def __init__(self, config=None):
        """
        Initialize the Emotion Agent
        
        Args:
            config: EmotionConfig object (optional)
        """
        self.config = config or EmotionConfig()
        
        # Initialize sub-agents
        self.voice_agent = EmotionalAgent(
            model_path=self.config.get_voice_model_path()
        )
        self.text_agent = TextEmotionAnalyzer()
        self.fusion_agent = EmotionFusion(
            voice_model_path=self.config.get_voice_model_path()
        )
        
        print("✅ Emotion Agent initialized successfully!")
    
    def analyze_voice(self, audio_path):
        """
        Analyze emotion from voice/audio only
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Emotion prediction
        """
        return self.voice_agent.predict_emotion(audio_path)
    
    def analyze_text(self, text):
        """
        Analyze emotion from text only
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Emotion prediction
        """
        return self.text_agent.predict_emotion(text)
    
    def analyze_multimodal(self, audio_path, text):
        """
        Analyze emotion from both voice and text
        
        Args:
            audio_path (str): Path to audio file
            text (str): Transcript or text
            
        Returns:
            dict: Fused emotion prediction
        """
        return self.fusion_agent.predict_from_audio_and_text(audio_path, text)
    
    def analyze_conversation(self, audio_segments, text_segments):
        """
        Analyze emotions across a conversation
        
        Args:
            audio_segments (list): List of audio file paths
            text_segments (list): List of text segments
            
        Returns:
            list: Emotion predictions for each segment
        """
        results = []
        for audio, text in zip(audio_segments, text_segments):
            result = self.analyze_multimodal(audio, text)
            results.append(result)
        return results