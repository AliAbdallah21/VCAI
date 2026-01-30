// frontend/src/services/emotionService.js

export async function analyzeEmotion(audioFile) {
  const formData = new FormData();
  formData.append('audio', audioFile);
  
  const response = await fetch('/api/analyze-emotion', {
    method: 'POST',
    body: formData
  });
  
  const emotion = await response.json();
  return emotion;
}

// Use it in your component
const emotion = await analyzeEmotion(recordedAudio);
console.log(`User is feeling: ${emotion.emotion}`);
displayEmotionUI(emotion);