from collections import deque

class ConversationBuffer:
    def __init__(self, max_turns: int = 5):
        """
        Maintains a sliding window of the last `max_turns` conversation exchanges.
        Each 'turn' is a tuple or dictionary of (user_input, assistant_output).
        """
        self.buffer = deque(maxlen=max_turns)

    def add_interaction(self, user_text: str, assistant_text: str):
        self.buffer.append({"role": "user", "content": user_text})
        self.buffer.append({"role": "assistant", "content": assistant_text})

    def get_history(self) -> list:
        """Returns the raw list of dictionaries formatted for Ollama/OpenAI API."""
        return list(self.buffer)
        
    def get_history_string(self) -> str:
        """Returns a formatted string of the recent conversation history for prompt injection."""
        if not self.buffer:
            return "No previous context."
            
        history_str = ""
        for msg in self.buffer:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        return history_str

    def clear(self):
        self.buffer.clear()
