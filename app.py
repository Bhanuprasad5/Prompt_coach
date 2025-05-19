import json
import os

import streamlit as st
from rag_pipeline import PromptCoachRAG


# Initialize the RAG pipeline
@st.cache_resource
def load_rag_pipeline():
    # Get API key from environment variable or use None for demo mode
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return PromptCoachRAG(r"D:\Gen_AI_Krish Naik\Gen AI\Apps\prompt_coach_app\embeddings\chunk_embeddings.json", api_key=api_key)

# Set up the Streamlit app
st.set_page_config(
    page_title="Prompt Coach",
    page_icon="✨",
    layout="wide"
)

# App title and description
st.title("✨ Prompt Coach")
st.markdown("""
This application helps you improve your prompts for AI systems by analyzing them against best practices 
from Google's prompt engineering guide. Enter your prompt below to receive feedback and suggestions.
""")

# API key input in sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key (optional)", 
                           value=os.environ.get("OPENAI_API_KEY", ""), 
                           type="password",
                           help="Enter your OpenAI API key to enable live analysis. If not provided, the app will run in demo mode with placeholder responses.")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API key set! The app will use OpenAI's models for analysis.")
    else:
        st.info("No API key provided. Running in demo mode with placeholder responses.")
    
    st.divider()
    
    st.header("About Prompt Coach")
    st.markdown("""
    Prompt Coach uses a Retrieval-Augmented Generation (RAG) system to analyze your prompts against 
    Google's prompt engineering best practices.
    
    ### How it works:
    1. You input your raw prompt
    2. The app analyzes it against the Google guide
    3. You receive feedback and a refined version
    4. You can iterate and improve your prompt
    
    ### Key features:
    - Prompt analysis and feedback
    - Refined prompt suggestions
    - Explanations of changes
    - Relevant examples from the guide
    """)
    
    st.divider()
    st.markdown("Powered by OpenAI and Google's Prompt Engineering Guide")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Your Prompt")
    user_prompt = st.text_area(
        "Enter your prompt here:",
        height=200,
        placeholder="Example: Write a blog post about AI"
    )
    
    analyze_button = st.button("Analyze Prompt", type="primary")
    
    # Example prompts
    st.subheader("Example prompts to try:")
    example_prompts = [
        "Write a blog post about AI",
        "Summarize this document",
        "Create a marketing email for our new product",
        "Help me brainstorm ideas for my presentation"
    ]
    
    for i, example in enumerate(example_prompts):
        if st.button(f"Try Example {i+1}", key=f"example_{i}"):
            user_prompt = example
            st.session_state.user_prompt = example
            analyze_button = True

# Initialize the RAG pipeline
rag = load_rag_pipeline()

# Process the prompt when the button is clicked
if analyze_button and user_prompt:
    with st.spinner("Analyzing your prompt..."):
        # Get analysis from the RAG pipeline
        analysis_result = rag.analyze_prompt(user_prompt)
        
        # Store the result in session state
        st.session_state.analysis_result = analysis_result
        st.session_state.user_prompt = user_prompt

# Display the analysis results
with col2:
    st.header("Analysis & Suggestions")
    
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        
        # Display the refined prompt
        st.subheader("✅ Refined Prompt")
        st.text(f"\n{result['analysis']['refined_prompt']}\n")
        
        # Display the explanation
        st.subheader("📝 Explanation of Changes")
        st.markdown(result['analysis']['explanation'])
        
        # Display strengths and weaknesses
        st.subheader("💪 Strengths")
        for strength in result['analysis']['strengths']:
            st.markdown(f"- {strength}")
        
        st.subheader("🔍 Areas for Improvement")
        for weakness in result['analysis']['weaknesses']:
            st.markdown(f"- {weakness}")
        
        # Display relevant sections from the guide
        st.subheader("📚 Relevant Guide Sections")
        for i, section in enumerate(result['relevant_sections']):
            with st.expander(f"Section {i+1}: {section['title']}"):
                st.markdown(section['content'])

# Interactive feedback section
if 'analysis_result' in st.session_state:
    st.divider()
    st.header("Interactive Feedback")
    
    feedback_col1, feedback_col2 = st.columns([1, 1])
    
    with feedback_col1:
        st.subheader("Try Again")
        new_prompt = st.text_area(
            "Modify your prompt based on feedback:",
            value=st.session_state.user_prompt,
            height=150
        )
        
        if st.button("Analyze Updated Prompt", type="primary"):
            with st.spinner("Analyzing your updated prompt..."):
                # Get analysis from the RAG pipeline
                analysis_result = rag.analyze_prompt(new_prompt)
                
                # Update the result in session state
                st.session_state.analysis_result = analysis_result
                st.session_state.user_prompt = new_prompt
                
                # Rerun to update the UI
                st.rerun()
    
    with feedback_col2:
        st.subheader("Was this helpful?")
        col_yes, col_no = st.columns(2)
        
        with col_yes:
            if st.button("👍 Yes"):
                st.success("Thank you for your feedback!")
        
        with col_no:
            if st.button("👎 No"):
                st.text_area("How can we improve?", placeholder="Please share your suggestions...")
                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback!")
