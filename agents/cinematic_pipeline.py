
#LangGraph-Orchestrated Cinematic RAG Pipeline  (LLM-First Architecture)

from __future__ import annotations
import asyncio
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END  # type: ignore

from core.llm_client import LLMClient  # type: ignore
from memory.quote_db import QuoteDBManager  # type: ignore
from memory.vector_db import VectorDBManager  # type: ignore
from memory.short_term import ConversationBuffer  # type: ignore
from agents.memory_gatekeeper import MemoryGatekeeper  # type: ignore


# ──────────────────────────────────────────────────────────────
# Typed State
# ──────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    user_query:      str
    preferences:     List[Dict[str, Any]]
    chat_history:    List[Dict[str, str]]
    direct_answer:   str                        # LLM's direct answer
    selected_quote:  Optional[Dict[str, Any]]   # optional quote enhancement
    final_response:  str


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────
class CinematicPipeline:
    """
    LLM-first LangGraph pipeline.
    Call  `await pipeline.run(user_query)` from your application.
    """

    def __init__(
        self,
        llm:         LLMClient,
        quote_db:    QuoteDBManager,
        vector_db:   VectorDBManager,
        chat_buffer: ConversationBuffer,
        gatekeeper:  MemoryGatekeeper,
    ):
        self.llm         = llm
        self.quote_db    = quote_db
        self.vector_db   = vector_db
        self.chat_buffer = chat_buffer
        self.gatekeeper  = gatekeeper

        # Track used quotes to avoid repetition
        self._used_indices: list[int] = []

        self._graph = self._build_graph()

    # ── Graph ─────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(GraphState)

        g.add_node("load_context",    self._node_load_context)
        g.add_node("generate_answer", self._node_generate_answer)
        g.add_node("retrieve_quote",  self._node_retrieve_quote)
        g.add_node("blend_answer",    self._node_blend_answer)

        g.set_entry_point("load_context")
        
        # Optimize: Route to both in parallel to hide RAG search latency
        g.add_edge("load_context", "generate_answer")
        g.add_edge("load_context", "retrieve_quote")
        
        # Converge
        g.add_edge("generate_answer", "blend_answer")
        g.add_edge("retrieve_quote", "blend_answer")
        
        g.add_edge("blend_answer", END)

        return g.compile()

    # ── Node 1: load_context ──────────────────────────────────

    async def _node_load_context(self, state: GraphState) -> GraphState:
        """
        Parallel FAISS ops (both sync, run in threads):
          • Decay-RAG  →  long-term preferences
          • Buffer     →  last N chat turns
        """
        # Query both preferences and custom facts (personal_context)
        prefs, facts, history = await asyncio.gather(
            asyncio.to_thread(
                self.vector_db.search_with_decay, "preferences", state["user_query"], 2
            ),
            asyncio.to_thread(
                self.vector_db.search_with_decay, "personal_context", state["user_query"], 2
            ),
            asyncio.to_thread(self.chat_buffer.get_history),
        )
        
        all_memory = prefs + facts
        if all_memory:
            print(f"\033[93m[Retrieved Memory]: '{all_memory[0]['text'][:50]}...'\033[0m")

        return {**state, "preferences": all_memory, "chat_history": history}

    # ── Node 2: generate_answer ───────────────────────────────

    async def _node_generate_answer(self, state: GraphState) -> GraphState:
        """
        LLM generates a short, direct answer to the user's question.
        This is ALWAYS relevant because it's directly responding to the query.
        """
        # Build system prompt with preferences, treating them as truth.
        system = (
            "You are a helpful, concise, and friendly assistant. "
            "Answer the user's question in 1-2 SHORT sentences maximum. "
            "Be natural, conversational, and directly address what they're asking."
        )
        if state["preferences"]:
            prefs = "\n".join(f"- {p['text']}" for p in state["preferences"])
            system += f"\n\nCRITICAL USER FACTS (TREAT THESE AS ABSOLUTE TRUTH OVER YOUR INTERNAL KNOWLEDGE. OBEY THEM EXACTLY):\n{prefs}"

        # Build messages with conversation history
        messages = [{"role": "system", "content": system}]
        messages.extend(state["chat_history"])
        messages.append({"role": "user", "content": state["user_query"]})

        print("\n💬 [LLM] Generating direct answer...")
        answer = ""
        async for chunk in self.llm.generate_stream(messages, temperature=0.6):
            answer += chunk

        print(f"\033[92m  Direct answer: {answer}\033[0m")
        return {**state, "direct_answer": answer}

    # ── Node 3: retrieve_quote ────────────────────────────────

    async def _node_retrieve_quote(self, state: GraphState) -> GraphState:
        """
        Search for a relevant quote based on the user's query.
        This is OPTIONAL - if no good match, we just use the direct answer.
        """
        query = state["user_query"].lower()
        words = query.split()
        
        # Skip quote search for very short queries (< 4 words)
        if len(words) < 4:
            print("\n⏭️ [Query too short - skipping quote enhancement]")
            return {**state, "selected_quote": None}

        print("\n🔍 [RAG] Searching for relevant quote...")
        print(f"  Query: {state['user_query']}")
        
        candidates = await asyncio.to_thread(
            self.quote_db.search_quote,
            state["user_query"],
            top_k=3,
            used_indices=self._used_indices,
        )
        
        if not candidates:
            print("  No quotes found")
            return {**state, "selected_quote": None}

        best = candidates[0]
        
        # Threshold at 0.25 balances finding solid situational matches without forcing irrelevant bad quotes.
        MIN_SCORE = 0.25
        if best["composite"] < MIN_SCORE:
            print(
                f"\n⚠️ [Quote rejected - low relevance]\n"
                f"  Match: '{best['metadata']['text'][:60]}...'\n"
                f"  Score: {best['composite']} < {MIN_SCORE}"
            )
            return {**state, "selected_quote": None}
        
        meta = best["metadata"]
        print(
            f"\033[93m✨ [Quote found]: '{meta['text']}'\n"
            f"   — {meta['character']} ({meta['source']})\n"
            f"   Score: {best['composite']}\033[0m"
        )
        
        # Track usage
        self._used_indices.append(best["index"])
        if len(self._used_indices) > 5:
            self._used_indices.pop(0)
        
        return {**state, "selected_quote": best}

    # ── Node 4: blend_answer ──────────────────────────────────

    async def _node_blend_answer(self, state: GraphState) -> GraphState:
        """
        If a quote was found, blend it into the direct answer naturally.
        Otherwise, just return the direct answer.
        """
        if not state.get("selected_quote"):
            # No quote - use direct answer
            print("\n✅ [Using direct answer - no quote enhancement]")
            return {**state, "final_response": state["direct_answer"]}
        
        # Blend quote into answer
        quote_data = state["selected_quote"]
        assert quote_data is not None, "selected_quote should not be None at this point"
        meta = quote_data["metadata"]
        quote_text = meta["text"]
        source = meta["source"]
        direct_answer = state["direct_answer"]
        
        # Build preference context
        prefs_str = ""
        if state["preferences"]:
            prefs_str = "\n".join(f"- {p['text']}" for p in state["preferences"])
            prefs_str = f"\n\nCRITICAL USER FACTS (TREAT THESE AS ABSOLUTE TRUTH OVER YOUR INTERNAL KNOWLEDGE):\n{prefs_str}"

        prompt = (
            f"You have a direct answer and a cinematic quote. "
            f"Blend them into a single, conversational text response. You MUST keep the vast majority of the quote's literal original words, but you are allowed to slightly tweak the grammar, tense, or pronouns to make it flow seamlessly into your sentence.\n\n"
            f"BLENDING STRATEGY:\n"
            f"1. Create ONE highly natural, short, and punchy response (1-2 sentences MAXIMUM).\n"
            f"2. Seamless integration: DO NOT use quotation marks. The quote should literally merge into your own sentence grammatically.\n"
            f"3. MATCH A HUMAN TONE: Never sound like an AI robot. Do NOT use corporate or therapy-speak like 'building resilience', 'foundation for growth', 'embrace it', or 'regardless of background'. Be direct, casual, and cinematic.\n"
            f"4. NEVER explain, interpret, or justify the quote! Just weave the core phrase directly into the conversation naturally.\n"
            f"5. AT THE VERY END OF YOUR RESPONSE, always append ({meta.get('character', 'Unknown')} - {source}). Do NOT embed the source inside the sentence.\n\n"
            f"EXAMPLES OF MASTERFUL, SHORT BLENDING:\n\n"
            f"Direct: You need to stand up for yourself because no one else will.\n"
            f"Quote: I am the one who knocks!\n"
            f"✅ GOOD (Start): You have to be the one who knocks, so adopt that mindset and stand up for yourself, because no one else will. (Walter White - Breaking Bad)\n\n"
            f"Direct: It's okay to fail as long as you keep trying.\n"
            f"Quote: There is no shame in being weak. The shame is in staying weak.\n"
            f"✅ GOOD (Middle): It's okay to fail because there's no shame in being weak, but the real shame is just staying weak and giving up. (Sung Jin-Woo - Solo Leveling)\n\n"
            f"Direct: Don't give up on your dreams, it takes time.\n"
            f"Quote: Even if you are not ready for the day, it cannot always be night.\n"
            f"✅ GOOD (End): It takes time to achieve your dreams, so don't give up entirely; even if you aren't ready for the day, it can't always be night. (Gwendolyn - Spider-Verse)\n\n"
            f"{prefs_str}\n\n"
            f"Context from conversation:\n{state.get('chat_history', [])}\n\n"
            f"NOW BLEND:\n"
            f"Direct answer: {direct_answer}\n"
            f"Base Quote: {quote_text}\n"
            f"Source Info (to put at the VERY end): ({meta.get('character', 'Unknown')} - {source})\n\n"
            f"Your perfectly seamless, natural response:"
        )

        print("\n🎨 [LLM] Blending quote into answer...")
        blended = ""
        async for chunk in self.llm.generate_stream(
            [{"role": "user", "content": prompt}],
            temperature=0.4,
        ):
            blended += chunk

        print(f"\033[96m  Final: {blended}\033[0m")
        return {**state, "final_response": blended}

    # ── Public API ────────────────────────────────────────────

    async def run(self, user_query: str) -> str:
        initial: GraphState = {
            "user_query":     user_query,
            "preferences":    [],
            "chat_history":   [],
            "direct_answer":  "",
            "selected_quote": None,
            "final_response": "",
        }

        result: GraphState = await self._graph.ainvoke(initial)
        
        # Handle parallel graph output formatting safely
        final = ""
        if isinstance(result, dict) and "final_response" in result:
             final = str(result["final_response"])
        else:
             final = str(result.get("final_response", ""))

        self.chat_buffer.add_interaction(user_text=user_query, assistant_text=final)

        print("\n🕵️ (Gatekeeper scanning...)")
        asyncio.create_task(self.gatekeeper.evaluate_and_store(user_query))

        return final
