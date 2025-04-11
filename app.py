import streamlit as st
from utils.data_loader import load_data
from utils.emotion_detection import detect_emotions
from utils.recommender import preprocess_emotions, get_recommendations
#re

# Page configuration
st.set_page_config(
    page_title="Emotion Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Emotion color mapping
emotion_colors = {
    'Angry': '#FF0000',
    'Disgusted': '#8B4513',
    'Fearful': '#800080',
    'Happy': '#FFD700',
    'Neutral': '#808080',
    'Sad': '#1E90FF',
    'Surprised': '#FFA500'
}

# Load data
data = load_data()

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/style.css")

# Main UI
def main():
    """Main application interface"""
    st.markdown("""
        <div class="header">
            <h1>Emotion-Based Music Recommender</h1>
            <p class="subtitle">Discover songs that match your current mood</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        detection_time = st.slider("Detection duration (seconds)", 5, 30, 10)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app detects your emotions through your webcam and recommends music based on your mood.")
        st.markdown("---")
        st.markdown("Created with ❤️ using Streamlit")

    # Emotion detection section
    if st.button('🎤 Detect My Emotion & Recommend Music'):
        with st.spinner('Detecting emotions... Please look at your camera'):
            detected_emotions = detect_emotions(detection_time)
        
        if detected_emotions:
            processed_emotions = preprocess_emotions(detected_emotions)
            st.success("Emotions successfully detected!")
            
            # Display detected emotions
            st.markdown("### Detected Emotions")
            cols = st.columns(len(processed_emotions))
            for idx, emotion in enumerate(processed_emotions):
                with cols[idx]:
                    st.markdown(
                        f'<div class="emotion-pill" style="background-color: {emotion_colors.get(emotion, "#666")}">'
                        f'{emotion}</div>',
                        unsafe_allow_html=True
                    )
            
            # Get recommendations
            recommendations = get_recommendations(processed_emotions, data)
            
            # Display recommendations
            st.markdown("### Recommended Songs")
            for i, row in recommendations.iterrows():
                st.markdown(
                    f'<div class="song-card">'
                    f'<a href="{row["link"]}" target="_blank" class="song-link">{i+1}. {row["name"]}</a>'
                    f'<p class="artist">{row["artist"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.error("No emotions detected. Please try again.")

if __name__ == "__main__":
    main()