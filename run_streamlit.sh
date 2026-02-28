#!/bin/bash

# Run the Streamlit Chat Application (Stable Version)
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run streamlit_app.py
