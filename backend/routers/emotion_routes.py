from fastapi import APIRouter, UploadFile, File
from emotion import EmotionAgent

router = APIRouter()
emotion_agent = EmotionAgent()

@router.post("/analyze-emotion")
async def analyze_emotion(audio: UploadFile = File(...)):
    # Save uploaded audio
    audio_path = f"/tmp/{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    
    # Analyze emotion
    result = emotion_agent.analyze_voice(audio_path)
    
    return result