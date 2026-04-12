"""
Main entry point for InBox Lens.
Combines UI, Engine, and Prompts logic, managing application state.
"""

import streamlit as st
import logging
from core.engine import AirLLMEngine
from core.prompts import PROMPT_CONFIG
import ui.components as ui

# Model path constraint as defined
MODEL_PATH = r"E:\airLLM-proj\Models\TinyLlama-1.1B-Chat"

def initialize_engine():
    """
    Initializes the AirLLMEngine only once using Streamlit's session state.
    This prevents the model from reloading when switching tabs or interacting with UI toggles.
    """
    if 'engine' not in st.session_state:
        with st.spinner("Initializing AirLLM Engine (Loading sharded model from disk - this may take a moment)..."):
            engine = AirLLMEngine(model_path=MODEL_PATH)
            engine.load()
            st.session_state.engine = engine
            logging.info("Engine successfully initialized and stored in session_state.")

def main():
    # Render main page headers
    ui.render_header()
    
    # Initialize engine via session state
    try:
        initialize_engine()
    except Exception as e:
        st.error(f"Critical Application Error: Could not initialize model engine. Details: {e}")
        st.stop()
        
    engine = st.session_state.engine
    
    # Define and render the application tabs
    tab_labels = [
        "📝 Summarize", 
        "✅ To-Do Extract", 
        "🎭 Sentiment Gauge", 
        "✨ Draft Refiner", 
        "🔄 Tone Shift"
    ]
    tabs = st.tabs(tab_labels)
    
    # Inject dependencies into UI components
    with tabs[0]:
        ui.render_summarize_tab(engine, PROMPT_CONFIG)
        
    with tabs[1]:
        ui.render_todo_tab(engine, PROMPT_CONFIG)
        
    with tabs[2]:
        ui.render_sentiment_tab(engine, PROMPT_CONFIG)
        
    with tabs[3]:
        ui.render_draft_refiner_tab(engine, PROMPT_CONFIG)
        
    with tabs[4]:
        ui.render_tone_shift_tab(engine, PROMPT_CONFIG)

if __name__ == "__main__":
    main()
