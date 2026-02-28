import re
import json
from core.llm_client import LLMClient
from memory.vector_db import VectorDBManager

class MemoryGatekeeper:
    def __init__(self, llm_client: LLMClient, vector_db: VectorDBManager):
        self.llm = llm_client
        self.vector_db = vector_db
        
        # Fast pre-flight check triggers to avoid slow LLM calls on casual chatter
        # Only strong, explicit preference statements
        self.trigger_phrases = [
            r"remember( that)?",
            r"from now on",
            r"i (really |strongly )?prefer",
            r"i struggle with",
            r"my goal is",
            r"call me",
            r"never (call|say|do|use)",
            r"i (really |absolutely )?(love|hate|can't stand)",
            r"always (use|call|refer)",
            r"my (name|favorite|passion) is"
        ]
        self.trigger_regex = re.compile('|'.join(self.trigger_phrases), re.IGNORECASE)

        self.system_prompt = (
            "You are an internal Memory Routing Agent. The user has provided a statement. "
            "Evaluate it VERY STRICTLY for long-term storage.\n\n"
            "ONLY STORE if the user is:\n"
            "1. Explicitly stating a strong preference or rule (e.g., 'I prefer X', 'always call me Y')\n"
            "2. Sharing important personal information (e.g., 'my name is X')\n"
            "3. Setting a permanent goal or learning objective\n"
            "4. Explicitly instructing you to remember a specific fact, claim, or worldview (e.g., 'Remember that X is Y')\n\n"
            "DO NOT STORE:\n"
            "- Casual mentions or passing comments (e.g., mentioning food, activities in passing)\n"
            "- Temporary states (e.g., 'I'm hungry', 'I'm tired')\n"
            "- Questions or jokes\n"
            "- Reactions to the assistant's responses\n\n"
            "OUTPUT STRICTLY VALID JSON matching this format:\n"
            '{"store": true/false, "summary": "One sentence summary of fact", "type": "preferences/learning_progress/personal_context", "importance": 0.0-1.0}\n\n'
            "If unsure, set store to false."
        )

    def _fast_check(self, user_text: str) -> bool:
        """Regex check to see if we even need to invoke the LLM."""
        return bool(self.trigger_regex.search(user_text))

    async def evaluate_and_store(self, user_text: str):
        """
        Runs asynchronously. If it passes the regex check, uses the LLM to format it 
        and permanently store it in FAISS.
        """
        if not self._fast_check(user_text):
            # Safe to drop. It's casual chat.
            return
            
        print("\n🧠 [Gatekeeper] Detected potential long-term memory. Scanning...")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"User said: {user_text}"}
        ]
        
        # Quick non-stream call
        response = await self.llm.generate_response(messages, temperature=0.1)
        
        try:
            # Clean up the output in case the LLM wrapped it in markdown code blocks
            clean_json = response.replace("```json", "").replace("```", "").strip()
            decision = json.loads(clean_json)
            
            if decision.get("store") is True:
                summary = decision.get("summary", "")
                domain = decision.get("type", "personal_context")
                importance = decision.get("importance", 0.5)
                
                # Verify domain validity
                if domain not in ["preferences", "learning_progress", "personal_context"]:
                    domain = "personal_context"
                    
                # Store permanently into FAISS
                self.vector_db.add_memory(
                    domain_name=domain,
                    text=summary,
                    importance=importance,
                    mem_type=domain
                )
        except Exception as e:
            print(f"⚠️ [Gatekeeper] Failed to parse memory JSON: {e}\nRaw output: {response}")
