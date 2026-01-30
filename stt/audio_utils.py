from emotion import EmotionAgent
from persona import PersonaAgent
from memory import MemoryAgent

class AgentOrchestrator:
    def __init__(self):
        self.emotion_agent = EmotionAgent()     # Your emotional agent
        self.persona_agent = PersonaAgent()
        self.memory_agent = MemoryAgent()
    
    def process_user_input(self, audio_path, transcript):
        # 1. Analyze emotion
        emotion = self.emotion_agent.analyze_voice(audio_path)
        
        # 2. Use emotion to adjust persona response
        persona_response = self.persona_agent.respond(
            transcript, 
            user_emotion=emotion['emotion']
        )
        
        # 3. Store in memory with emotion context
        self.memory_agent.store(
            transcript, 
            emotion=emotion['emotion']
        )
        
        return {
            'emotion': emotion,
            'response': persona_response
        }