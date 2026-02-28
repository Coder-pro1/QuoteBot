#  QuoteBot

QuoteBot  AI chat assistant is integrated with famous movie/tv-series/anime quotes and catchphrases using a specialized dual-database RAG pipeline. ALso it uses LangGraph to orchestrate multiple specialized AI agents.
![QuoteBot UI](/demo.png)

##  Core Architecture

```
* Local LLM*            : Ollama (Qwen 3.1)
* Agent Orchestration   : LangGraph
* Local Vector DB       : FAISS
* Embedding Model       : Hugging Face Sentence-Transformers (`all-MiniLM-L6-v2`)
* Text-to-Speech        : Hugging Face / gTTS
* Frontend / UI         : Streamlit
```
##  What Makes QuoteBot Different?
Unlike basic RAG systems that retrieve literal text matches, QuoteBot uses a situational pipeline and a dual layer memory allowing it to remember context and recent user interactions as well.

* **Dual-Layer Memory:** Unlike flat RAG databases, Quotes and User Facts are maintained in entirely separate FAISS indices, ensuring the LLM never hallucinates movie quotes as real-world facts.
* **Situational Indexing:** Instead of embedding raw quotes directly, it indexes synthesized "Situational Usecases" to match your conversational context and emotional intent rather than just text keywords.
* **Zero Latency Generation:** To mask vector database latency, QuoteBot uses LangGraph to execute LLM response generation and FAISS retrieval in parallel, concurrent threads.
* **Background Memory:** An asynchronous Memory Gatekeeper listens for user facts in the background, continuously updating a Time-Decayed personal FAISS index.

## 💡 Functionality
 Contextual Memory,Seamless Quote Integration,Voice Synthesis

## 🚀 Setup & Execution 

1. **Prerequisites**: Ensure [Ollama](https://ollama.com/) is installed and running locally. Pull your preferred model (Default is `qwen3:8b`).
2. **Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3. **Run Application**:
    ```bash
    sh run_streamlit.sh
    # OR natively: 
    streamlit run streamlit_app.py
    ```
