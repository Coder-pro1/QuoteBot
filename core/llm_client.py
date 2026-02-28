import httpx
import json
from core.config import OLLAMA_BASE_URL, DEFAULT_MODEL

class LLMClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{self.base_url}/api/chat"

    async def generate_response(self, messages: list, temperature: float = 0.7) -> str:
        """
        Non-streaming response (waits for full output).
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "")
                return str(content)
            except Exception as e:
                print(f"Error calling Ollama: {e}")
                return ""
        return ""  # explicit fallback for type checker

    async def generate_stream(self, messages: list, temperature: float = 0.7):
        """
        Streaming response, yields tokens as they are generated to feel much faster.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream("POST", self.api_url, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if not chunk: # empty chunk
                            continue
                        
                        data = json.loads(chunk)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                            
            except Exception as e:
                print(f"Error streaming from Ollama: {e}")
                yield ""
