"""
Streamlit Chat Interface - Cinematic RAG Chat
==============================================
Chat with AI powered by movie quotes and catchphrases
"""
import streamlit as st
import asyncio
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from core.llm_client import LLMClient
from memory.quote_db import QuoteDBManager
from memory.vector_db import VectorDBManager
from memory.short_term import ConversationBuffer
from agents.cinematic_pipeline import CinematicPipeline
from agents.memory_gatekeeper import MemoryGatekeeper
from core.tts_engine import get_tts_engine


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────
def sanitize_for_tts(text: str) -> str:
    """
    Remove attribution, symbols, emojis, and brackets from the response so the TTS
    engine reads it naturally without vocalizing punctuation or markdown.
    """
    # 1. Remove the explicit (Source) trailing at the end
    text = re.sub(r'\s*\([^)]+\)\s*\.?\s*$', '', text)
    
    # 2. Remove any other content inside brackets or parentheses
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    
    # 3. Remove asterisks, underscores, and tildes (markdown formatting)
    text = re.sub(r'[*_~]', '', text)
    
    # 4. Remove emojis and visual noise (keep only alphanumeric and readable punctuation)
    text = re.sub(r'[^\w\s.,!?:;\'"—“”‘’\-]', '', text)
    
    # 5. Clean up duplicate spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ─────────────────────────────────────────────────────────────
# Initialize Pipeline (with caching)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def init_pipeline():
    """Initialize the cinematic pipeline (cached to avoid reloading)."""
    llm = LLMClient()
    quote_db = QuoteDBManager(
        json_path="data/quote_dictionary.json",
        index_path="data/indexes/quote.index",
    )
    vector_db = VectorDBManager(
        data_dir="data",
        index_dir="data/indexes",
        model_name="all-MiniLM-L6-v2",
    )
    buffer = ConversationBuffer(max_turns=5)
    gatekeeper = MemoryGatekeeper(llm, vector_db)
    pipeline = CinematicPipeline(
        llm=llm,
        quote_db=quote_db,
        vector_db=vector_db,
        chat_buffer=buffer,
        gatekeeper=gatekeeper,
    )
    return pipeline, buffer


# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="QuoteBot",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* User messages - right aligned, blue bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            flex-direction: row-reverse;
            text-align: right;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-radius: 20px 20px 5px 20px !important;
            padding: 15px 20px !important;
            margin-left: 20% !important;
            margin-right: 10px !important;
            color: white !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
            color: white !important;
        }
        
        /* Assistant messages - left aligned, dark bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
            border-radius: 20px 20px 20px 5px !important;
            padding: 15px 20px !important;
            margin-right: 20% !important;
            margin-left: 10px !important;
            color: white !important;
        }
        
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p {
            color: white !important;
        }
        
        /* Chat input styling */
        .stChatInput {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 25px;
        }
        
        .stTextInput > div > div > input {
            border-radius: 25px !important;
            background: rgba(255, 255, 255, 0.9);
        }
        
        /* Header styling */
        h1 {
            background: linear-gradient(45deg, #FFD700, #FF6B6B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2C3E50 0%, #34495e 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Audio player styling */
        audio {
            width: 100%;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header with cinematic flair
    st.markdown("<h1> QuoteBot </h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2em; color: #FFD700;'>"
        " <i>Powered by Movie/Anime Quotes & Catchphrases</i> "
        "</p>", 
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "pipeline" not in st.session_state:
        with st.spinner("⏳ Loading cinematic pipeline..."):
            pipeline, buffer = init_pipeline()
            st.session_state.pipeline = pipeline
            st.session_state.buffer = buffer
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")
        
        # Voice output settings
        st.markdown("### 🔊 Voice Output")
        enable_tts = st.checkbox(
            "🎙️ Enable Text-to-Speech", 
            value=False, 
            help="Hear responses spoken aloud"
        )
        
        if enable_tts:
            st.success("✨ Voice enabled!")
            st.caption("Attribution will be removed from voice output")
        
        st.markdown("---")
        
        # Memory management
        st.markdown("### 💾 Conversation")
        
        # Show conversation stats
        if st.session_state.messages:
            message_count = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.metric("Messages", message_count)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.buffer.clear()
                st.success("✅ Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("🧹 Reset Prefs", use_container_width=True, help="Clear learned preferences"):
                import os
                try:
                    # Remove preference indexes
                    pref_index = "data/indexes/preferences.index"
                    pref_meta = "data/indexes/preferences_meta.json"
                    if os.path.exists(pref_index):
                        os.remove(pref_index)
                    if os.path.exists(pref_meta):
                        os.remove(pref_meta)
                    st.success("✅ Preferences reset!")
                    st.info("Restart the app to apply changes")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Tips
        st.markdown("### 💡 Tips")
        st.info(
            "🎯 **Try asking:**\n"
            "- What's your purpose?\n"
            "- How to deal with failure?\n"
            "- I need motivation\n"
            "- Hello!"
        )
    
    # Display chat history with enhanced styling
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div style="color: white;">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🎬"):
                st.markdown(f'<div style="color: white;">{message["content"]}</div>', unsafe_allow_html=True)
                
                # Play audio if available
                if "audio" in message and message["audio"]:
                    try:
                        audio_format = "audio/mp3" if message["audio"].endswith('.mp3') else "audio/wav"
                        st.audio(message["audio"], format=audio_format)
                    except:
                        pass
    
    # Chat input with placeholder
    user_input = st.chat_input(
        "🎬 Type your message... (e.g., 'I need motivation')",
        key="chat_input"
    )
    
    if user_input:
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div style="color: white;">{user_input}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.chat_message("assistant", avatar="🎬"):
            with st.spinner("🎭 Finding the perfect quote..."):
                # Run async pipeline
                response = asyncio.run(st.session_state.pipeline.run(user_input))
                
                # Display response with formatting
                st.markdown(f'<div style="color: white;">{response}</div>', unsafe_allow_html=True)
                
                # Generate TTS if enabled
                audio_file = None
                if enable_tts:
                    with st.spinner("🔊 Generating voice..."):
                        try:
                            tts_engine = get_tts_engine()
                            # Remove attribution (character/source) and sanitize symbols before TTS
                            tts_text = sanitize_for_tts(response)
                            audio_file = tts_engine.generate_speech(tts_text)
                            if audio_file:
                                # Determine audio format
                                audio_format = "audio/mp3" if audio_file.endswith('.mp3') else "audio/wav"
                                st.audio(audio_file, format=audio_format)
                        except Exception as e:
                            st.error(f"⚠️ TTS error: {e}")
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "audio": audio_file
        })
        
        st.rerun()


if __name__ == "__main__":
    main()
