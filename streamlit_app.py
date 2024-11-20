import streamlit as st
import numpy as np
from openai import OpenAI
from Levenshtein import distance, jaro, jaro_winkler
from scipy.spatial.distance import cosine
from difflib import SequenceMatcher
import textdistance
import time

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text):
    """Get embedding from OpenAI API"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    return 1 - cosine(embedding1, embedding2)

def normalize_levenshtein(s1, s2):
    """Calculate normalized Levenshtein distance"""
    lev_dist = distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1 - (lev_dist / max_len)

def get_color(similarity_score):
    """Return a color based on similarity score"""
    if similarity_score >= 0.8:
        return "#4CAF50"  # Green
    elif similarity_score >= 0.6:
        return "#8BC34A"  # Light Green
    elif similarity_score >= 0.4:
        return "#FFEB3B"  # Yellow
    elif similarity_score >= 0.2:
        return "#FFC107"  # Amber
    else:
        return "#F44336"  # Red

# Set page config
st.set_page_config(
    page_title="Semantik",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better visualization
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .metric-container {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    .similarity-label {
        font-size: 1.2em;
        font-weight: bold;
        color: #E0E0E0;
    }
    .similarity-score {
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }
    .stTextArea textarea {
        background-color: #2D2D2D;
        color: white;
        border: 1px solid #404040;
    }
    .explanation {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title with logo
st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0;">🔤 Semantik</h1>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<p style='color: #B0B0B0;'>Advanced String Similarity Analysis Tool</p>", unsafe_allow_html=True)

# Create two text input fields
col1, col2 = st.columns(2)
with col1:
    string1 = st.text_area("Enter first string:", height=100)
with col2:
    string2 = st.text_area("Enter second string:", height=100)

if string1 and string2:
    st.subheader("Similarity Analysis")
    
    # Create a progress placeholder
    progress_placeholder = st.empty()
    
    with st.spinner("Computing similarity measures..."):
        # Initialize similarities dict
        similarities = {}
        
        # Show progress
        progress_bar = progress_placeholder.progress(0)
        
        # Calculate various similarity measures with progress updates
        similarities["Levenshtein"] = normalize_levenshtein(string1, string2)
        progress_bar.progress(20)
        
        similarities["Jaro"] = jaro(string1, string2)
        progress_bar.progress(40)
        
        similarities["Jaro-Winkler"] = jaro_winkler(string1, string2)
        progress_bar.progress(50)
        
        similarities["Ratcliff/Obershelp"] = SequenceMatcher(None, string1, string2).ratio()
        progress_bar.progress(60)
        
        similarities["Jaccard"] = textdistance.jaccard.normalized_similarity(string1, string2)
        progress_bar.progress(70)
        
        similarities["Sorensen-Dice"] = textdistance.sorensen.normalized_similarity(string1, string2)
        progress_bar.progress(80)
        
        # Calculate embedding similarity
        try:
            embedding1 = get_embedding(string1)
            embedding2 = get_embedding(string2)
            similarities["Semantic (OpenAI)"] = cosine_similarity(embedding1, embedding2)
            progress_bar.progress(100)
            
            # Remove progress bar
            progress_placeholder.empty()
            
            # Create three columns for metrics
            cols = st.columns(3)
            
            # Display all similarities with color coding
            for idx, (method, score) in enumerate(similarities.items()):
                col = cols[idx % 3]
                with col:
                    st.markdown(f"""
                    <div class="metric-container" style="border-left: 5px solid {get_color(score)}">
                        <div class="similarity-label">{method}</div>
                        <div class="similarity-score" style="color: {get_color(score)}">
                            {score:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add explanations
            st.markdown("""
            <div class="explanation">
                <h3>Understanding the Metrics</h3>
                <p>All metrics are normalized to range from 0 (completely different) to 1 (identical):</p>
                <ul>
                    <li><strong>Levenshtein:</strong> Based on the minimum number of single-character edits needed</li>
                    <li><strong>Jaro:</strong> Accounts for character matchings and transpositions</li>
                    <li><strong>Jaro-Winkler:</strong> Modified Jaro that gives higher scores to strings matching from the beginning</li>
                    <li><strong>Ratcliff/Obershelp:</strong> Based on the number of matching characters in sequence</li>
                    <li><strong>Jaccard:</strong> Measures similarity based on character n-gram overlap</li>
                    <li><strong>Sorensen-Dice:</strong> Similar to Jaccard but gives more weight to matches</li>
                    <li><strong>Semantic:</strong> Uses OpenAI embeddings to measure meaning similarity</li>
                </ul>
                <p>🎨 <strong>Color Legend:</strong></p>
                <ul>
                    <li style="color: #4CAF50;">■ Green (≥0.8): Very similar</li>
                    <li style="color: #8BC34A;">■ Light green (0.6-0.8): Similar</li>
                    <li style="color: #FFEB3B;">■ Yellow (0.4-0.6): Moderately similar</li>
                    <li style="color: #FFC107;">■ Amber (0.2-0.4): Somewhat different</li>
                    <li style="color: #F44336;">■ Red (<0.2): Very different</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error calculating similarities: {str(e)}")
else:
    st.info("Please enter two strings to compare.")

# Setup instructions in sidebar
st.sidebar.title("About Semantik")
st.sidebar.write("""
This tool provides comprehensive string similarity analysis using multiple algorithms.
""")