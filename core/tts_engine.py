import tempfile
import os
from typing import Optional
from gtts import gTTS

class TTSEngine:
    def __init__(self):
        print("💡 TTS engine initialized (gTTS)")
    
    def generate_speech(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            output_path = temp_file.name
            temp_file.close()
        
        try:
            if output_path.endswith('.wav'):
                output_path = output_path.replace('.wav', '.mp3')
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            return output_path
        except Exception as e:
            print(f"❌ gTTS failed: {e}")
            return None

_tts_engine = None

def get_tts_engine() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine
