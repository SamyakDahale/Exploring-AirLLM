"""
UI Components for InBox Lens.
Reusable Streamlit UI functions to keep main.py clean and modular.
"""

import streamlit as st

def render_header():
    """Renders the top level configuration and header for the app."""
    st.set_page_config(page_title="InBox Lens", layout="wide", page_icon="🔍")
    st.title("🔍 InBox Lens")
    st.markdown("Powered by **AirLLM** and **TinyLlama-1.1B** *(Offline & Privacy-Preserving)*")
    st.markdown("---")

def render_summarize_tab(engine, prompt_config):
    st.header("📝 Email Summarizer")
    st.markdown("Extract a concise 3-bullet point summary from long email threads.")
    email_body = st.text_area("Paste the email body here:", height=200, key="summarize_text")
    
    if st.button("Summarize", type="primary", key="btn_summarize"):
        if not email_body.strip():
            st.warning("Please enter an email body first.")
            return
            
        with st.spinner("Analyzing email..."):
            system_message = prompt_config["summarize"]["system_message"]
            user_prompt = f"Email:\n{email_body}"
            summary = engine.generate_response(system_message, user_prompt)
            
        st.success("Summary Generated!")
        st.markdown(summary)

def render_todo_tab(engine, prompt_config):
    st.header("✅ To-Do Extractor")
    st.markdown("Find deadlines and actionable tasks buried in the text.")
    email_body = st.text_area("Paste the email body here:", height=200, key="todo_text")
    
    if st.button("Extract Action Items", type="primary", key="btn_todo"):
        if not email_body.strip():
            st.warning("Please enter an email body first.")
            return
            
        with st.spinner("Extracting tasks..."):
            system_message = prompt_config["todo_extraction"]["system_message"]
            user_prompt = f"Email:\n{email_body}"
            tasks = engine.generate_response(system_message, user_prompt)
            
        st.success("Extraction Complete!")
        st.markdown(tasks)

def render_sentiment_tab(engine, prompt_config):
    st.header("🎭 Sentiment Gauge")
    st.markdown("Detect the 'vibe' and overall tone of the sender.")
    email_body = st.text_area("Paste the email body here:", height=200, key="sentiment_text")
    
    if st.button("Analyze Sentiment", type="primary", key="btn_sentiment"):
        if not email_body.strip():
            st.warning("Please enter an email body first.")
            return
            
        with st.spinner("Gauging sentiment..."):
            system_message = prompt_config["sentiment_analysis"]["system_message"]
            user_prompt = f"Email:\n{email_body}"
            # Less tokens needed for a categorical sentiment output
            sentiment = engine.generate_response(system_message, user_prompt, max_new_tokens=100) 
            
        st.success("Analysis Complete!")
        st.markdown(f"**Result:**\n\n{sentiment}")

def render_draft_refiner_tab(engine, prompt_config):
    st.header("✨ Draft Refiner")
    st.markdown("Expand rough bullet points into a full, formatted email draft.")
    draft_body = st.text_area("Paste your rough bullet points here:", height=200, key="refiner_text")
    
    if st.button("Refine Draft", type="primary", key="btn_refiner"):
        if not draft_body.strip():
            st.warning("Please enter rough points first.")
            return
            
        with st.spinner("Refining draft..."):
            system_message = prompt_config["draft_refiner"]["system_message"]
            user_prompt = f"Rough Notes:\n{draft_body}"
            refined = engine.generate_response(system_message, user_prompt)
            
        st.success("Draft Refined!")
        st.text_area("Refined Output (Ready to Copy):", value=refined, height=250)

def render_tone_shift_tab(engine, prompt_config):
    st.header("🔄 Tone Shifter")
    st.markdown("Rewrite a rough drafted email to match a specific audience tone.")
    draft_body = st.text_area("Paste your rough draft here:", height=200, key="tone_text")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        target_tone = st.selectbox("Select Target Tone:", ["Professional", "Urgent", "Friendly", "Concise", "Persuasive"], key="tone_select")
    
    if st.button("Rewrite Draft", type="primary", key="btn_tone"):
        if not draft_body.strip():
            st.warning("Please enter a rough draft first.")
            return
            
        with st.spinner(f"Rewriting draft in a {target_tone.lower()} tone..."):
            system_message = prompt_config["tone_shift"]["system_message_template"].format(tone=target_tone)
            user_prompt = f"Draft:\n{draft_body}"
            rewritten = engine.generate_response(system_message, user_prompt)
            
        st.success("Draft Rewritten!")
        st.text_area("Rewritten Output (Ready to Copy):", value=rewritten, height=250)
